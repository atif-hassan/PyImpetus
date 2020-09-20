import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
import scipy.stats as ss
import copy
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor




# CURRENTLY THIS ONLY SUPPORTS BINARY CLASSIFICATION
class inter_IAMB:
    def __init__(self, model=None, min_feat_proba_thresh=0.1, p_val_thresh=0.05, k_feats_select=5, num_simul=100, stratified=False, num_cv_splits=5, regression=False, verbose=1):
        # Defines the model which will be used for conditional feature selection
        if model:
            self.model = model
        elif regression:
            self.model = DecisionTreeRegressor(random_state=27)
        else:
            self.model = DecisionTreeClassifier(random_state=27)
        # The minimum probability of occurrence that a feature should possess over all folds for it to be considered in the final MB
        self.min_feat_proba_thresh = min_feat_proba_thresh
        # The p-value below which the feature will considered as a candidate for the final MB
        self.p_val_thresh = p_val_thresh
        # The number of features to select during growth phase of InterIAMB algorithm
        self.k_feats_select = k_feats_select
        # Number of train-test splits to perform to check usefulness of each feature
        self.num_simul = num_simul
        # CV should be stratified or simple
        self.stratified = stratified
        # Number of folds for CV 
        self.num_cv_splits = num_cv_splits
        # Defines the task whether it is regression or classification
        self.regression = regression
        # For printing some stuff
        self.verbose = verbose

        # The final list of features
        self.final_feats = list()




    def CPI(self, X, Y, Z, B, orig_model, regression):
        # Generate the data matrix
        # Always keep the variable to check, in front
        X = np.reshape(X, (-1, 1))
        if Z is None:
            data = X
        elif Z.ndim == 1:
            data = np.concatenate((X, np.reshape(Z, (-1, 1))), axis=1)
        else:
            data = np.concatenate((X, Z), axis=1)

        # testX -> prediction on correct set
        # testY -> prediction on permuted set
        testX, testY = list(), list()

        for i in range(B):
            x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=i)
            model = copy.deepcopy(orig_model)
            model.fit(x_train, y_train)
            if regression:
                preds = model.predict(x_test)
                testX.append(mean_squared_error(y_test, preds))
            else:
                preds = model.predict_proba(x_test)[:,1]
                testX.append(log_loss(y_test, preds))

            # Perform permutation
            np.random.seed(i)
            np.random.shuffle(x_train[:,0])
            np.random.shuffle(x_test[:,0])

            model = copy.deepcopy(orig_model)
            model.fit(x_train, y_train)
            if regression:
                preds = model.predict(x_test)
                testY.append(mean_squared_error(y_test, preds))
            else:
                preds = model.predict_proba(x_test)[:,1]
                testY.append(log_loss(y_test, preds))

        # Since we want log_loss to be lesser for testX,
        # we therefore perform one tail paired t-test (to the left and not right)
        t_stat, p_val = ss.ttest_ind(testX, testY, nan_policy='omit')
        return ss.t.cdf(t_stat, len(testX) + len(testY) - 2) # Left part of t-test



       
    # Function that performs the growth stage of the Inter-IAMB algorithm
    def grow(self, data, orig_data, Y, MB, n, B, model, regression):
        best = list()
        # For each feature in MB, check if it is false positive
        for col in data.columns:
            if len(MB) > 0:
                p_val = self.CPI(data[col].values, Y, orig_data[MB].values, B, model, regression)
            else:
                p_val = self.CPI(data[col].values, Y, None, B,  model, regression)
            best.append([col, p_val])
        
        # Sort and pick the top n features
        best.sort(key = lambda x: x[1])
        return best[:n]




    # Function that performs the shrinking stage of the Inter-IAMB algorithm
    def shrink(self, data, Y, MB, thresh, B, model, regression):
        # Reverse the MB and shrink since the first feature in the list is the most important
        MB.reverse()
        # If there is only one element in the MB, no shrinkage is required
        if len(MB) < 2:
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
            p_val = self.CPI(data[col].values, Y, data[MB_to_consider].values, B, model, regression)
            if p_val > thresh:
                remove.append(col)
        # Reverse back the list
        MB.reverse()
        return remove




    # Function that performs the Inter-IAMB algorithm
    # Outputs the Markov Blanket (MB) as well as the false positive features
    def inter_IAMB(self, data, Y):
        # Keep a copy of the original data
        orig_data = data.copy()
        Y = np.reshape(Y, (-1, 1))
        MB = list()
        
        # Run the algorithm until there is no change in current epoch MB and previous epoch MB
        while True:
            old_MB = list(MB)
            
            # Growth phase
            best = self.grow(data, orig_data, Y, MB, self.k_feats_select, self.num_simul, self.model, self.regression)
            for best_feat, best_val in best:
                if best_val < self.p_val_thresh:
                    MB.append(best_feat)

            # Shrink phase
            remove_feats = self.shrink(orig_data, Y, MB, self.p_val_thresh, self.num_simul, self.model, self.regression)
            for feat in remove_feats:
                MB.pop(MB.index(feat))
            
            # Remove all features in MB and remove_feats from the dataframe
            feats_to_remove = list()
            for i in MB+remove_feats:
                if i in data.columns:
                    feats_to_remove.append(i)
            data = data.drop(feats_to_remove, axis=1)
            
            # Break if current MB is same as previous MB
            if old_MB == MB:
                break
            print("Candidate features: ", MB)
            
        # Finally, return the Markov Blanket of the target variable
        return MB




    # Call this to get a list of selected features
    def fit(self, data, target):
        # List of candidate features from each fold
        feature_sets = list()

        # Go ahead with stratified, if specified. See, I even rhymed it for you
        if self.stratified:
            kfold = StratifiedKFold(n_splits=self.num_cv_splits, random_state=27, shuffle=True)
            for cv_index, (train, test) in enumerate(kfold.split(data, data[target])):
                if self.verbose == 1:
                    print("CV Number: ", cv_index+1, "\n#############################")
                    
                # Define the X and Y variables
                X, Y = data.loc[train], data[target].values[train]
                X = X.drop([target], axis=1)

                # The inter_IAMB function returns a list of features for current fold
                feats = self.inter_IAMB(X.copy(), Y)
                feature_sets.extend(feats)

                # Do some printing if specified
                if self.verbose == 1:
                    print("\nFinal features selected in this fold: ", feats)
                    print()

        # Otherwise the usual KFold
        else:
            kfold = KFold(n_splits=self.num_cv_splits, random_state=27, shuffle=True)
            for cv_index, (train, test) in enumerate(kfold.split(data)):
                if self.verbose == 1:
                    print("CV Number: ", cv_index+1, "\n#############################")
                    
                # Define the X and Y variables
                X, Y = data.loc[train], data[target].values[train]
                X = X.drop([target], axis=1)

                # The inter_IAMB function returns a list of features for current fold
                feats = self.inter_IAMB(X.copy(), Y)
                feature_sets.extend(feats)

                # Do some printing if specified
                if self.verbose == 1:
                    print("\nFinal features selected in this fold: ", feats)
                    print()

        # Get the list of all candidate features and their probabilities
        proposed_feats = [[i, j/self.num_cv_splits] for i, j in Counter(feature_sets).items()]
        # Pretty printing
        if self.verbose == 1:
            print("\n\nFINAL SELECTED FEATURES\n##################################")
        # Select only those candidate features that have a probability higher than min_feat_proba_thresh
        for a, b in proposed_feats:
            if b > self.min_feat_proba_thresh:
                if self.verbose == 1:
                    print("Feature: ", a, "\tProbability Score: ", b)
                self.final_feats.append(a)
        # Finally, return the final set of features
        return self.final_feats




    # Call this function to simply return the pruned (feature-wise) data
    def transform(self, data):
        return data[self.final_feats]




    # A quick wrapper function for fit() and transform()
    def fit_transform(self, data, target):
        # First call fit
        self.fit(data, target)
        # Now transform the data
        return self.fit_transform(data)
