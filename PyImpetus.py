import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
import scipy.stats as ss
import copy
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection._split import check_cv
from sklearn.utils.validation import check_is_fitted, check_random_state
from joblib import Parallel, delayed


class inter_IAMB(TransformerMixin, BaseEstimator):
    """
    PyImpetus - interIAMB feature selection algorithm with the conditional mutual
    information part being replaced by a conditional test.

    PyImpetus is a feature selection algorithm that picks features by considering their performance
    both individually as well as conditioned on other selected features. This allows the algorithm
    to not only select the best set of features, it also selects the best set of features
    that play well with each other. For example, the best performing feature might not
    play well with others while the remaining features, when taken together could out-perform
    the best feature. PyImpetus takes this into account and produces the best possible combination.

    The fitting process may take considerable time, depending on dataset size and the amount of features.
    Lowering the ``num_simul`` will reduce runtime at the expense of accuracy, though the bigger
    the dataset, the less simulations are required to obtain a good result.

    Parameters
    ----------
    model : estimator object, default=None
        The model which will be used for conditional feature selection. If ``None``,
        will use ``DecisionTreeRegressor`` or ``DecisionTreeClassifier`` depending on the
        ``regression`` param.

    min_feat_proba_thresh : float, default=0.1
        The minimum probability of occurrence that a feature should possess over all
        folds for it to be considered in the final MB.

    p_val_thresh : float, default=0.05
        The p-value below which the feature will considered as a candidate for the final MB.

    k_feats_select : int, default=5
        The number of features to select during growth phase of InterIAMB algorithm.

    num_simul : int, default=100
        Number of train-test splits to perform to check usefulness of each feature.
        Has a major effect on runtime.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the ``regression`` param is False and ``y`` is
        either binary or multiclass, StratifiedKFold is used. In all
        other cases, KFold is used.

    regression : bool, default=False
        Defines the task - whether it is regression or classification.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState instance, default=None
        Pass an int for reproducible output across multiple function calls.

    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Attributes
    ----------
    final_feats_ : list
        Final list of column indices selected.

    final_feats_pandas_ : list
        Final list of pandas DataFrame column names selected, corresponding
        to ``final_feats_``. If fitted ``X`` was not a pandas DataFrame, will
        be ``None``.

    X_cols_number_ : int
        Number of columns in fitted data.

    Notes
    -----
    PyImpetus is basically the interIAMB algorithm as provided in the paper
    "An Improved IAMB Algorithm for Markov Blanket Discovery" [1]_ with the conditional mutual information
    part being replaced by a conditional test. This test is as described in the paper
    "Testing Conditional Independence in Supervised Learning Algorithms" [2]_.

    References
    ----------
    .. [1] Zhang, Y., Zhang, Z., Liu, K., & Qian, G. (2010).
       "An Improved IAMB Algorithm for Markov Blanket Discovery". JCP, 5(11), 1755-1761.

    .. [2] Watson, D. S., & Wright, M. N. (2019). "Testing Conditional Independence in
       Supervised Learning Algorithms". arXiv preprint arXiv:1901.09917.
    """

    def __init__(
        self,
        model=None,
        min_feat_proba_thresh=0.1,
        p_val_thresh=0.05,
        k_feats_select=5,
        num_simul=100,
        cv=5,
        regression=False,
        random_state=None,
        verbose=0,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
    ):
        # Defines the model which will be used for conditional feature selection
        self.random_state = random_state

        if model:
            self.model = clone(model)
        elif regression:
            self.model = DecisionTreeRegressor()
        else:
            self.model = DecisionTreeClassifier()
        # The minimum probability of occurrence that a feature should possess over all folds for it to be considered in the final MB
        self.min_feat_proba_thresh = min_feat_proba_thresh
        # The p-value below which the feature will considered as a candidate for the final MB
        self.p_val_thresh = p_val_thresh
        # The number of features to select during growth phase of InterIAMB algorithm
        self.k_feats_select = k_feats_select
        # Number of train-test splits to perform to check usefulness of each feature
        self.num_simul = num_simul
        # Number of folds for CV
        self.cv = cv
        # For printing some stuff
        self.verbose = verbose
        # Defines the task whether it is regression or classification
        self.regression = regression

        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch

    def _CPI(self, X, y, Z, B, orig_model, regression):
        # Generate the data matrix
        # Always keep the variable to check, in front
        if Z is None:
            data = X
        else:
            data = np.concatenate((np.reshape(X, (-1, 1)), Z), axis=1)
        if data.ndim == 1:
            data = np.reshape(data, (-1, 1))

        # testX -> prediction on correct set
        # testY -> prediction on permuted set
        testX, testY = list(), list()

        for i in range(B):
            x_train, x_test, y_train, y_test = train_test_split(
                data, y, test_size=0.2, random_state=self.rng_
            )
            model = clone(orig_model)
            model.fit(x_train, y_train)
            if regression:
                preds = model.predict(x_test)
                testX.append(mean_squared_error(y_test, preds))
            else:
                preds = model.predict_proba(x_test)[:, 1]
                testX.append(log_loss(y_test, preds))

            # Perform permutation
            self.rng_.shuffle(x_train[:, 0])
            self.rng_.shuffle(x_test[:, 0])

            model = clone(orig_model)
            model.fit(x_train, y_train)
            if regression:
                preds = model.predict(x_test)
                testY.append(mean_squared_error(y_test, preds))
            else:
                preds = model.predict_proba(x_test)[:, 1]
                testY.append(log_loss(y_test, preds))

        # Since we want log_loss to be lesser for testX,
        # we therefore perform one tail paired t-test (to the left and not right)
        t_stat, p_val = ss.ttest_ind(testX, testY, nan_policy="omit")
        return ss.t.cdf(t_stat, len(testX) + len(testY) - 2)  # Left part of t-test

    # Function that performs the growth stage of the Inter-IAMB algorithm
    def _grow(self, X, orig_X, y, MB, n, B, model, regression):
        best = list()
        # For each feature in MB, check if it is false positive
        for col in range(X.shape[1]):
            p_val = self._CPI(
                X[:, col],
                y,
                orig_X[:, MB] if len(MB) > 0 else None,
                B,
                model,
                regression,
            )
            best.append([col, p_val])

        # Sort and pick the top n features
        best.sort(key=lambda x: x[1])
        return best[:n]

    # Function that performs the shrinking stage of the Inter-IAMB algorithm
    def _shrink(self, X, y, MB, thresh, B, model, regression):
        # Reverse the MB and shrink since the first feature in the list is the most important
        MB.reverse()
        # Generate a list for false positive features
        remove = list()
        # If there is only one element in the MB, no shrinkage is required
        if len(MB) < 2:
            return remove
        # For each feature in MB, check if it is false positive
        for col in MB:
            MB_to_consider = [i for i in MB if i not in [col] + remove]
            # Again, if there is only one element in MB, no shrinkage is required
            if len(MB_to_consider) < 1:
                break
            # Get the p-value from the Conditional Predictive Information Test
            p_val = self._CPI(X[:, col], y, X[:, MB_to_consider], B, model, regression)
            if p_val > thresh:
                remove.append(col)
        # Reverse back the list
        MB.reverse()
        return remove

    # Function that performs the Inter-IAMB algorithm
    # Outputs the Markov Blanket (MB) as well as the false positive features
    def _inter_IAMB(self, X, y, X_cols):
        # Keep a copy of the original data
        orig_X = X.copy()
        MB = list()

        # Run the algorithm until there is no change in current epoch MB and previous epoch MB
        while True:
            old_MB = MB.copy()
            # Growth phase
            best = self._grow(
                X,
                orig_X,
                y,
                MB,
                self.k_feats_select,
                self.num_simul,
                self.model,
                self.regression,
            )
            for best_feat, best_val in best:
                if best_val < self.p_val_thresh:
                    MB.append(best_feat)
            # Shrink phase
            remove_feats = self._shrink(
                orig_X,
                y,
                MB,
                self.p_val_thresh,
                self.num_simul,
                self.model,
                self.regression,
            )
            for feat in remove_feats:
                MB.remove(feat)

            # Remove all features in MB and remove_feats from the dataframe
            feats_to_remove = MB + remove_feats
            feats_to_keep = [
                x for x in range(0, X.shape[1]) if x not in feats_to_remove
            ]
            X = X[:, feats_to_keep]

            # Break if current MB is same as previous MB
            if old_MB == MB:
                break
            if self.verbose >= 1:
                print("Candidate features: ", self._translate_columns(MB, X_cols))

        # Finally, return the Markov Blanket of the target variable
        return MB

    def _fit_single(self, X, y, train, cv_index, X_cols):
        if self.verbose >= 1:
            print("CV Number: ", cv_index + 1, "\n#############################")

        # Define the X and Y variables
        X_, y_ = X[train], y[train]

        # The inter_IAMB function returns a list of features for current fold
        feats = self._inter_IAMB(X_, y_, X_cols)

        # Do some printing if specified
        if self.verbose >= 1:
            print(
                "\nFinal features selected in this fold: ",
                self._translate_columns(feats, X_cols),
            )
            print()
        return feats

    def _translate_columns(self, X_idx, X_cols):
        if X_cols is not None:
            try:
                r = [X_cols[x] for x in X_idx]
            except TypeError:
                r = X_cols[X_idx]
            if len(r) == 1:
                return r[0]
            return r
        return X_idx

    # Call this to get a list of selected features
    def fit(self, X, y, groups=None):
        """
        Fit the feature selector.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        
        y : array-like, shape (n_samples, )
            Target relative to X for classification or regression.
        
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" ``cv``
            instance (e.g., ``~sklearn.model_selection.GroupKFold``).
        
        Returns
        -------
        self
            object
        """
        # if not isinstance(X, pd.DataFrame):
        #    raise TypeError("X param must be a pandas.DataFrame instance")

        # if not (isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)):
        #    raise TypeError(
        #        "y param must be a pandas.DataFrame or pandas.Series instance"
        #    )

        # The final list of features
        self.final_feats_ = list()

        self.rng_ = check_random_state(self.random_state)
        self.model.set_params(random_state=self.rng_)

        cv = check_cv(self.cv, y, classifier=(not self.regression))
        num_cv_splits = cv.get_n_splits(X, y, groups)
        try:
            X_cols = list(X.columns)
        except:
            X_cols = None

        self.X_cols_number_ = X.shape[1]

        try:
            X = X.to_numpy()
            y = y.to_numpy()
        except:
            X = np.array(X)
            y = np.array(y)

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
        )
        feature_sets = parallel(
            delayed(self._fit_single)(X, y, train_test_tuple[0], cv_index, X_cols)
            for cv_index, train_test_tuple in enumerate(cv.split(X, y, groups))
        )
        # flatten the list
        print(feature_sets)
        feature_sets = [item for sublist in feature_sets for item in sublist]
        # Get the list of all candidate features and their probabilities
        proposed_feats = [
            [i, j / num_cv_splits] for i, j in Counter(feature_sets).items()
        ]
        # Pretty printing
        if self.verbose >= 1:
            print("\n\nFINAL SELECTED FEATURES\n##################################")
        # Select only those candidate features that have a probability higher than min_feat_proba_thresh
        self.final_feats_ = set()
        for a, b in proposed_feats:
            if b > self.min_feat_proba_thresh:
                if self.verbose >= 1:
                    print(
                        "Feature: ",
                        self._translate_columns(a, X_cols),
                        "\tProbability Score: ",
                        b,
                    )
                self.final_feats_.add(a)
        self.final_feats_ = sorted(list(self.final_feats_))
        self.final_feats_pandas_ = (
            self._translate_columns(self.final_feats_, X_cols)
            if X_cols is not None
            else None
        )
        # Finally, return the final set of features
        return self

    # Call this function to simply return the pruned (feature-wise) data
    def transform(self, X, y=None):
        """
        Transform data by selecting features.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        
        y : Not used.
        
        groups : Not used.
        
        Returns
        -------
        Xt : pandas.DataFrame
            Transformed data.
        """
        check_is_fitted(self)
        X = X.copy()
        if isinstance(X, pd.DataFrame) and self.final_feats_pandas_ is not None:
            return X[self.final_feats_pandas_]
        if X.shape[1] != self.X_cols_number_:
            raise ValueError(f"X has {X.shape[1]} columns, expected {self.X_cols_number_}.")
        try:
            return X.iloc[:, self.final_feats_]
        except:
            return X[:, self.final_feats_]

    # A quick wrapper function for fit() and transform()
    def fit_transform(self, X, y, groups=None):
        """
        Fit the feature selector and transform data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        
        y : array-like, shape (n_samples, )
            Target relative to X for classification or regression.
        
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" ``cv``
            instance (e.g., ``~sklearn.model_selection.GroupKFold``).
        
        Returns
        -------
        Xt : pandas.DataFrame
            Transformed data.
        """
        # First call fit
        self.fit(X, y, groups=groups)
        # Now transform the data
        return self.transform(X)
