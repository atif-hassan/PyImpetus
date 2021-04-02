import math
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.metrics import log_loss, mean_squared_error
import scipy.stats as ss
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextlib
import joblib



# tqdm enabled for joblib
# This is all thanks to the answer by frenzykryger at https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()   










class PPIMBC(TransformerMixin, BaseEstimator):
    def __init__(self, model=None, p_val_thresh=0.05, num_simul=30, cv=0, random_state=None, n_jobs=-1, verbose=2):
        self.random_state = random_state
        if model is not None:
            self.model = model
        else:
            self.model = DecisionTreeClassifier(random_state=self.random_state)
        self.p_val_thresh = p_val_thresh
        self.num_simul = num_simul
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.MB = None
        self.feat_imp_scores = list()





    #---------------------------------------------------------------------------------------------
    # _feature_importance() - Function that gives score with and without feature transformation #
    #---------------------------------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    # i - a number for shuffling the data. Based on the number of simulations provided by user
    def _feature_importance(self, data, Y, i):
        # The target variable comes in 2D shape. Reshape it
        Y = np.ravel(Y)
        # Split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=i)
        # Find all the labels. If there are more than 2 labels, then it multi-class classification
        # So handle accordingly
        self_labels = np.unique(y_train)
        if len(self_labels) <= 2:
            self_labels = None
        # Clone the user provided model
        model = clone(self.model)
        # Fit the model on train data
        model.fit(x_train, y_train)
        # SVM and RidgeClassifiers dont have predict_proba functions so handle accordingly
        if "SVC" in type(model).__name__ or "RidgeClassifier" in type(model).__name__:
            preds = model.decision_function(x_test)
        else:
            preds = model.predict_proba(x_test)
        # Calculate log_loss metric. T-test is perfomed on this value
        x = log_loss(y_test, preds, labels=self_labels)

        np.random.seed(i)
        np.random.shuffle(x_test[:,0])

        # SVM and RidgeClassifiers dont have predict_proba functions so handle accordingly
        if "SVC" in type(model).__name__ or "RidgeClassifier" in type(model).__name__:
            preds = model.decision_function(x_test)
        else:
            preds = model.predict_proba(x_test)
        # Calculate log_loss metric. T-test is perfomed on this value
        y = log_loss(y_test, preds, labels=self_labels)

        return [x, y]




    #-----------------------------------------------------
    # _PPI() - Function that provides feature importance #
    #-----------------------------------------------------
    # X - The feature to provide importance for
    # Y - The target variable
    # Z - The remaining variables upon which X is conditionally tested
    def _PPI(self, X, Y, Z, col):
        X = np.reshape(X, (-1, 1))
        # Either growth stage or shrink stage
        if Z is None:
            data = X
        elif Z.ndim == 1:
            # Always keep the variable to check, in front
            data = np.concatenate((X, np.reshape(Z, (-1, 1))), axis=1)
        else:
            data = np.concatenate((X, Z), axis=1)

        # testX -> prediction on correct set
        # testY -> prediction on permuted set
        testX, testY = list(), list()
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        p_values = parallel(delayed(self._feature_importance)(data, Y, i) for i in range(self.num_simul))
        p_values = np.array(p_values)
        testX, testY = p_values[:,0], p_values[:,1]

        # Since we want log_loss to be lesser for testX,
        # we therefore perform one tail paired t-test (to the left and not right)
        t_stat, p_val = ss.ttest_ind(testX, testY, alternative="less")
        if col is None:
            return p_val#ss.t.cdf(t_stat, len(testX) + len(testY) - 2) # Left part of t-test
        else:
            return [col, p_val]#ss.t.cdf(t_stat, len(testX) + len(testY) - 2)] # Left part of t-test





    #---------------------------------------------------------------------
    # _grow() - Function that performs the growth stage of the algorithm #
    #---------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def _grow(self, data, Y):
        MB = list()
        # For each feature find its individual importance in relation to the target variable
        # Since each feature's importance is checked individually, we can therefore run them in parallel
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        feats_and_pval = parallel(delayed(self._PPI)(data[col].values, Y, None, col) for col in data.columns)
        
        # Sort the features according to their importance (decreasing order)
        feats_and_pval.sort(key = lambda x: x[1], reverse=True)
        # Only significant features are added the MB
        for feat, p_val in feats_and_pval:
            if p_val < self.p_val_thresh:
                MB.append(feat)
        return MB





    #-----------------------------------------------------------------------
    # _shrink() - Function that performs the shrink stage of the algorithm #
    #-----------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    # MB - The MB currently populated by features from growth stage
    def _shrink(self, data, Y, MB):
        scores = list()
        # If MB is empty, return
        if len(MB) < 1:
            return list()
        # Generate a list for false positive features
        remove = list()
        # For each feature in MB, check if it is false positive
        for col in MB:
            MB_to_consider = [i for i in MB if i not in [col]+remove]
            # Again, if there is only one element in MB, no shrinkage is required
            if len(MB_to_consider) < 1:
                break
            # Get the p-value from the Predictive Permutation Independence Test
            p_val = self._PPI(data[col].values, Y, data[MB_to_consider].values, None)
            if p_val > self.p_val_thresh:
                remove.append(col)
            else:
                scores.append(np.log(1/p_val))
        # Finally, return only those features that are not present in MB
        return [i for i in MB if i not in remove], scores


    #----------------------------------------------------------------------
    # _find_MB() - Function that finds the MB of provided target variable #
    #----------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def _find_MB(self, data, Y):
        # Keep a copy of the original data
        orig_data = data.copy()
        # The target variable needs to be reshaped for downstream processing
        Y = np.reshape(Y, (-1, 1))

        # Growth phase
        MB = self._grow(data, Y)

        # Shrink phase. This is the final MB
        MB, scores = self._shrink(orig_data, Y, MB)

        return MB, scores




    #------------------------------------------------------------------------------------------
    # fit() - Function that finds the MB of target variable using CV strategy for aggregation #
    #------------------------------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def fit(self, data, Y):
        # for a value of 0, no CV is applied
        if self.cv!= 0:
            # Find MB for each fold in parallel
            parallel = Parallel(n_jobs=self.n_jobs)#, verbose=self.verbose)
            if type(self.cv).__name__ == "StratifiedKFold":
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv.get_n_splits())) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data, Y))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data, Y))
            elif type(self.cv).__name__ == "KFold":
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv.get_n_splits())) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data))
            else:
                kfold = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv)) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data))

            # Separate out the features from the importance scores
            for i in range(len(tmp)):
                if i == 0:
                    feature_sets, scores = [tmp[i][0]], [tmp[i][1]]
                else:
                    feature_sets.append(tmp[i][0])
                    scores.append(tmp[i][1])
                
            # Now get all the features and find their frequencies
            final_feats = dict()
            for fs in feature_sets:
                for i in fs:
                    if i not in final_feats:
                        final_feats[i] = 1
                    else:
                        final_feats[i]+= 1

            # Now find the most robust MB
            final_MB, max_score, final_feat_imp = list(), 0, list()
            for fs, feat_imp in zip(feature_sets, scores):
                tmp = [final_feats[i] for i in fs]
                score = sum(tmp)/max(len(tmp), 1)
                if score > max_score:
                    final_MB = fs
                    final_feat_imp = feat_imp
                    max_score = score
                    
            self.MB = final_MB
            self.feat_imp_scores = final_feat_imp

        else:
            self.MB, self.feat_imp_scores = self._find_MB(data.copy(), Y)


    

    #--------------------------------------------------------------
    # transform() - Function that returns the MB part of the data #
    #--------------------------------------------------------------
    def transform(self, data):
        return data[self.MB]




    #-----------------------------------------------------------
    # fit_transform() - Wrapper function for fit and transform #
    #-----------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def fit_transform(self, data, Y):
        self.fit(data, Y)
        return self.transform(data)




    #-------------------------------------------------------------------------
    # feature_importance() - Wrapper function for plotting feature importance #
    #--------------------------------------------------------------------------
    def feature_importance(self):
        y_axis = np.arange(len(self.MB))
        x_axis = self.feat_imp_scores

        sns.barplot(x=x_axis, y=y_axis, orient="h")
        plt.yticks(y_axis, [str(i) for i in self.MB], size='small')
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        sns.despine()
        plt.show()
        














class PPIMBR(TransformerMixin, BaseEstimator):
    def __init__(self, model=None, p_val_thresh=0.05, num_simul=30, cv=0, random_state=None, n_jobs=-1, verbose=2):
        self.random_state = random_state
        if model is not None:
            self.model = model
        else:
            self.model = DecisionTreeRegressor(random_state=self.random_state)
        self.p_val_thresh = p_val_thresh
        self.num_simul = num_simul
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.MB = None
        self.feat_imp_scores = list()





    #---------------------------------------------------------------------------------------------
    # _feature_importance() - Function that gives score with and without feature transformation #
    #---------------------------------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    # i - a number for shuffling the data. Based on the number of simulations provided by user
    def _feature_importance(self, data, Y, i):
        # The target variable comes in 2D shape. Reshape it
        Y = np.ravel(Y)
        # Clone the user provided model
        model = clone(self.model)
        # Split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=i)

        # Fit the model on train data
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        # Calculate rmse metric. T-test is perfomed on this value
        x = mean_squared_error(y_test, preds)

        # Perform transformation
        np.random.seed(i)
        np.random.shuffle(x_test[:,0])

        preds = model.predict(x_test)
        # Calculate mse metric. T-test is perfomed on this value
        y = mean_squared_error(y_test, preds)

        return [x, y]





    #-----------------------------------------------------
    # _CPI() - Function that provides feature importance #
    #-----------------------------------------------------
    # X - The feature to provide importance for
    # Y - The target variable
    # Z - The remaining variables upon which X is conditionally tested
    def _PPI(self, X, Y, Z, col):
        X = np.reshape(X, (-1, 1))
        # Either growth stage or shrink stage
        if Z is None:
            data = X
        elif Z.ndim == 1:
            # Always keep the variable to check, in front
            data = np.concatenate((X, np.reshape(Z, (-1, 1))), axis=1)
        else:
            data = np.concatenate((X, Z), axis=1)

        # testX -> prediction on correct set
        # testY -> prediction on permuted set
        testX, testY = list(), list()
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        p_values = parallel(delayed(self._feature_importance)(data, Y, i) for i in range(self.num_simul))
        p_values = np.array(p_values)
        testX, testY = p_values[:,0], p_values[:,1]

        # Since we want log_loss to be lesser for testX,
        # we therefore perform one tail paired t-test (to the left and not right)
        t_stat, p_val = ss.ttest_ind(testX, testY, nan_policy='omit', alternative="less")
        if col is None:
            return p_val#ss.t.cdf(t_stat, len(testX) + len(testY) - 2) # Left part of t-test
        else:
            return [col, p_val]#ss.t.cdf(t_stat, len(testX) + len(testY) - 2)] # Left part of t-test





    #---------------------------------------------------------------------
    # _grow() - Function that performs the growth stage of the algorithm #
    #---------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def _grow(self, data, Y):
        MB = list()
        # For each feature find its individual importance in relation to the target variable
        # Since each feature's importance is checked individually, we can therefore run them in parallel
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        feats_and_pval = parallel(delayed(self._PPI)(data[col].values, Y, None, col) for col in data.columns)
        #feats_and_pval = [self._CPI(data[col].values, Y, None, col) for col in tqdm(data.columns)]
        
        # Sort the features according to their importance (decreasing order)
        feats_and_pval.sort(key = lambda x: x[1], reverse=True)
        # Only significant features are added the MB
        for feat, p_val in feats_and_pval:
            if p_val < self.p_val_thresh:
                MB.append(feat)
        return MB





    #-----------------------------------------------------------------------
    # _shrink() - Function that performs the shrink stage of the algorithm #
    #-----------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    # MB - The MB currently populated by features from growth stage
    def _shrink(self, data, Y, MB):
        scores = list()
        # If MB is empty, return
        if len(MB) < 1:
            return list()
        # Generate a list for false positive features
        remove = list()
        # For each feature in MB, check if it is false positive
        for col in MB:
            MB_to_consider = [i for i in MB if i not in [col]+remove]
            # Again, if there is only one element in MB, no shrinkage is required
            if len(MB_to_consider) < 1:
                break
            # Get the p-value from the Conditional Predictive Information Test
            p_val = self._PPI(data[col].values, Y, data[MB_to_consider].values, None)
            if p_val > self.p_val_thresh:
                remove.append(col)
            else:
                scores.append(np.log(1/p_val))
        # Finally, return only those features that are not present in MB
        return [i for i in MB if i not in remove], scores


    #----------------------------------------------------------------------
    # _find_MB() - Function that finds the MB of provided target variable #
    #----------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def _find_MB(self, data, Y):
        # Keep a copy of the original data
        orig_data = data.copy()
        # The target variable needs to be reshaped for downstream processing
        Y = np.reshape(Y, (-1, 1))

        # Growth phase
        MB = self._grow(data, Y)

        # Shrink phase. This is the final MB
        MB, scores = self._shrink(orig_data, Y, MB)

        return MB, scores




    #------------------------------------------------------------------------------------------
    # fit() - Function that finds the MB of target variable using CV strategy for aggregation #
    #------------------------------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def fit(self, data, Y):
        # for a value of 0, no CV is applied
        if self.cv!= 0:
            # Find MB for each fold in parallel
            parallel = Parallel(n_jobs=self.n_jobs)#, verbose=self.verbose)
            if type(self.cv).__name__ == "StratifiedKFold":
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv.get_n_splits())) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data, Y))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data, Y))
            elif type(self.cv).__name__ == "KFold":
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv.get_n_splits())) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in self.cv.split(data))
            else:
                kfold = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
                if self.verbose > 0:
                    with tqdm_joblib(tqdm(desc="Progress bar", total=self.cv)) as progress_bar:
                        tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data))
                else:
                    tmp = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data))

            # Separate out the features from the importance scores
            for i in range(len(tmp)):
                if i == 0:
                    feature_sets, scores = [tmp[i][0]], [tmp[i][1]]
                else:
                    feature_sets.append(tmp[i][0])
                    scores.append(tmp[i][1])
                
            # Now get all the features and find their frequencies
            final_feats = dict()
            for fs in feature_sets:
                for i in fs:
                    if i not in final_feats:
                        final_feats[i] = 1
                    else:
                        final_feats[i]+= 1

            # Now find the most robust MB
            final_MB, max_score, final_feat_imp = list(), 0, list()
            for fs, feat_imp in zip(feature_sets, scores):
                tmp = [final_feats[i] for i in fs]
                score = sum(tmp)/max(len(tmp), 1)
                if score > max_score:
                    final_MB = fs
                    final_feat_imp = feat_imp
                    max_score = score
                    
            self.MB = final_MB
            self.feat_imp_scores = final_feat_imp

        else:
            self.MB, self.feat_imp_scores = self._find_MB(data.copy(), Y)


    

    #--------------------------------------------------------------
    # transform() - Function that returns the MB part of the data #
    #--------------------------------------------------------------
    def transform(self, data):
        return data[self.MB]




    #-----------------------------------------------------------
    # fit_transform() - Wrapper function for fit and transform #
    #-----------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def fit_transform(self, data, Y):
        self.fit(data, Y)
        return self.transform(data)




    #-------------------------------------------------------------------------
    # feature_importance() - Wrapper function for plotting feature importance #
    #--------------------------------------------------------------------------
    def feature_importance(self):
        y_axis = np.arange(len(self.MB))
        x_axis = self.feat_imp_scores

        sns.barplot(x=x_axis, y=y_axis, orient="h")
        plt.yticks(y_axis, [str(i) for i in self.MB], size='small')
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        sns.despine()
        plt.show()
