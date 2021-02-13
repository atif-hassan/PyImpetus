import math
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.metrics import log_loss
import scipy.stats as ss
from collections import Counter






class CPIMB(TransformerMixin, BaseEstimator):
    def __init__(self, model=None, p_val_thresh=0.05, num_simul=30, cv=5, random_state=None, n_jobs=-1, verbose=0):
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
    # _CPI() - Function that provides feature importance #
    #-----------------------------------------------------
    # X - The feature to provide importance for
    # Y - The target variable
    # Z - The remaining variables upon which X is conditionally tested
    def _CPI(self, X, Y, Z, col):
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
        t_stat, p_val = ss.ttest_ind(testX, testY, nan_policy='omit')
        if col is None:
            return ss.t.cdf(t_stat, len(testX) + len(testY) - 2) # Left part of t-test
        else:
            return [col, ss.t.cdf(t_stat, len(testX) + len(testY) - 2)] # Left part of t-test





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
        feats_and_pval = parallel(delayed(self._CPI)(data[col].values, Y, None, col) for col in data.columns)
        
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
            p_val = self._CPI(data[col].values, Y, data[MB_to_consider].values, None)
            if p_val > self.p_val_thresh:
                remove.append(col)
        # Finally, return only those features that are not present in MB
        return [i for i in MB if i not in remove]


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
        MB = self._shrink(orig_data, Y, MB)

        return MB




    #------------------------------------------------------------------------------------------
    # fit() - Function that finds the MB of target variable using CV strategy for aggregation #
    #------------------------------------------------------------------------------------------
    # data - The data provided by user
    # Y - Target variable
    def fit(self, data, Y):
        # Find MB for each fold in parallel
        kfold = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        feature_sets = parallel(delayed(self._find_MB)(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data))
        #feature_sets = [self._find_MB(data.iloc[train].copy(), Y[train]) for train, test in kfold.split(data)]
            
        # Now get all the features and find their frequencies
        final_feats = dict()
        for fs in feature_sets:
            for i in fs:
                if i not in final_feats:
                    final_feats[i] = 1
                else:
                    final_feats[i]+= 1

        # Now find the most robust MB
        final_MB, max_score = list(), 0
        for fs in feature_sets:
            tmp = [final_feats[i] for i in fs]
            score = sum(tmp)/max(len(tmp), 1)
            if score > max_score:
                final_MB = fs
                max_score = score
                
        self.MB = final_MB


    

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

    
