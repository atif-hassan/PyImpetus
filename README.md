[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/PyImpetus.svg)](https://pypi.python.org/pypi/PyImpetus/)
[![Downloads](https://pepy.tech/badge/PyImpetus)](https://pepy.tech/project/PyImpetus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/PyImpetus/commits/master)
# PyImpetus
PyImpetus is a Markov Blanket based **feature selection algorithm** that selects a subset of features by considering their performance both individually as well as a group. This allows the algorithm to not only select the best set of features, but also select the **best set of features that play well with each other**. For example, the best performing feature might not play well with others while the remaining features, when taken together could out-perform the best feature. PyImpetus takes this into account and produces the best possible combination. Thus, the algorithm provides a minimal feature subset. So, **you do not have to decide how many features to take. PyImpetus selects the optimal set for you.**

PyImpetus has been completely revamped and now supports **binary classification, multi-class classification and regression** tasks. It uses a novel CV based aggregation method to recommend the most roubst set of minimal features (Markov Blanket).

PyImpetus was tested on 13 datasets and outperformed state-of-the-art Markov Blanket learning algorithms on all of them along with traditional feature selection algorithms such as Forward Feature Selection, Backward Feature Elimination and Recursive Feature Elimination.

## How to install?
```pip install PyImpetus```

## Functions and parameters
```python
# The initialization of PyImpetus takes in multiple parameters as input
# PPIMBC is for classification
model = PPIMBC(model, p_val_thresh, num_simul, cv, verbose, random_state, n_jobs)
```
- **model** - `estimator object, default=DecisionTreeClassifier()` The model which is used to perform classification in order to find feature importance via significance-test. 
- **p_val_thresh** - `float, default=0.05` The p-value (in this case, feature importance) below which a feature will be considered as a candidate for the final MB.
- **num_simul** - `int, default=10` **(This feature has huge impact on speed)** Number of train-test splits to perform to check usefulness of each feature. For large datasets, the value should be considerably reduced though do not go below 5.
- **cv** - `cv object/int, default=5` Determines the the number of splits for cross-validation. Sklearn CV object can also be passed.
- **verbose** - `int, default=0` Controls the verbosity: the higher, more the messages.
- **random_state** - `int or RandomState instance, default=None` Pass an int for reproducible output across multiple function calls.
- **n_jobs** - `int, default=-1` The number of CPUs to use to do the computation.
	- `None` means 1 unless in a `:obj:joblib.parallel_backend` context.
	- `-1` means using all processors.

```python
# The initialization of PyImpetus takes in multiple parameters as input
# PPIMBC is for regression
model = PPIMBR(model, p_val_thresh, num_simul, cv, verbose, random_state, n_jobs)
```
- **model** - `estimator object, default=DecisionTreeRegressor()` The model which is used to perform regression in order to find feature importance via significance-test. 
- **p_val_thresh** - `float, default=0.05` The p-value (in this case, feature importance) below which a feature will be considered as a candidate for the final MB.
- **num_simul** - `int, default=10` **(This feature has huge impact on speed)** Number of train-test splits to perform to check usefulness of each feature. For large datasets, the value should be considerably reduced though do not go below 5.
- **cv** - `cv object/int, default=5` Determines the the number of splits for cross-validation. Sklearn CV object can also be passed.
- **verbose** - `int, default=0` Controls the verbosity: the higher, more the messages.
- **random_state** - `int or RandomState instance, default=None` Pass an int for reproducible output across multiple function calls.
- **n_jobs** - `int, default=-1` The number of CPUs to use to do the computation.
	- `None` means 1 unless in a `:obj:joblib.parallel_backend` context.
	- `-1` means using all processors.

```python
# To fit PyImpetus on provided dataset and find recommended features
fit(data, target)
```
- **data** - A pandas dataframe upon which feature selection is to be applied
- **target** - A numpy array, denoting the target variable

```python
# This function returns the names of the columns that form the MB (These are the recommended features)
transform(data)
```
- **data** - A pandas dataframe which needs to be pruned

```python
# To fit PyImpetus on provided dataset and return pruned data
fit_transform(data, target)
```
- **data** - A pandas dataframe upon which feature selection is to be applied
- **target** - A numpy array, denoting the target variable

```python
# To plot XGBoost style feature importance
feature_importance()
```


## How to import?
```python
from PyImeptus import PPIMBC, PPIMBR
```

## Usage
```python
# Import the algorithm. PPIMBC is for classification and PPIMBR is for regression
from PyImeptus import PPIMBC, PPIMBR
# Initialize the PyImpetus object
model = PPIMBC(model=SVC(random_state=27, class_weight="balanced"), p_val_thresh=0.05, num_simul=30, cv=5, random_state=27, n_jobs=-1, verbose=2)
# The fit_transform function is a wrapper for the fit and transform functions, individually.
# The fit function finds the MB for given data while transform function provides the pruned form of the dataset
df_train = model.fit_transform(df_train.drop("Response", axis=1), df_train["Response"].values)
df_test = model.transform(df_test)
# Check out the MB
print(model.MB)
# Check out the feature importance scores for the selected feature subset
print(model.feat_imp_scores)
# Get a plot of the feature importance scores
model.feature_importance()
```

## For better accuracy
- Increase the **cv** value
- Increase the **num_simul** value
- Use non-linear models for feature selection

## For better speeds
- Decrease the **cv** value. For large datasets cv might not be required. Therefore, set **cv=0** to disable the aggregation step. This will result in less robust feature subset selection but at much faster speeds
- Decrease the **num_simul** value but don't decrease it below 5
- Set **n_jobs** to -1
- Use linear models

## For selection of less features
- Try reducing the **p_val_thresh** value

## Performance in terms of Accuracy (classification) and MSE (regression)
| Dataset | # of samples | # of features | Task Type | Score (all features) | Score (with PyImpetus) | Tutorial |
| --- | --- | --- | --- |--- |--- |--- |
| Ionosphere | 351 | 34 | Classification | 88.01 | 91.73 | [tutorial here](https://github.com/atif-hassan/PyImpetus/blob/master/tutorials/Classification_Tutorial.ipynb) |
| slice_localization_data | 53500 | 384 | Regression | 5.98 | 5.16 | [tutorial here](https://github.com/atif-hassan/PyImpetus/blob/master/tutorials/Regression_Tutorial.ipynb) |

Here, for the first task, a higher accuracy score is better while for the second, task, a lower MSE (Mean Squared Error) is better.

**Note:** Number of features selected by PyImpetus:
- For Ionosphere dataset: 5 (14% of features selected to achieve 3% improvement in Accuracy)
- For slice_location_data: 45 (11.7% of features selected to achieve 0.82 reduction in MSE)

## Performance in terms of Time (in seconds)
| Dataset | # of samples | # of features | Task Type | Time (all features) | Time (with PyImpetus) |
| --- | --- | --- | --- | --- | --- |
| Ionosphere | 351 | 34 | Classification | 0.0066 | 35.37 |
| slice_localization_data | 53500 | 384 | Regression | 5.49 | 1296.13 |

## Future Ideas
- Let me know

## Feature Request
Drop me an email at **atif.hit.hassan@gmail.com** if you want any particular feature

# Please cite this work as
Reference to the upcoming paper will be added here
