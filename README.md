[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/PyImpetus.svg)](https://pypi.python.org/pypi/PyImpetus/)
[![Downloads](https://pepy.tech/badge/PyImpetus)](https://pepy.tech/project/PyImpetus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/PyImpetus/commits/master)
# PyImpetus
PyImpetus is a Markov Blanket based **feature selection algorithm** that selects a subset of features by considering their performance both individually as well as a group. This allows the algorithm to not only select the best set of features, but also select the **best set of features that play well with each other**. For example, the best performing feature might not play well with others while the remaining features, when taken together could out-perform the best feature. PyImpetus takes this into account and produces the best possible combination. Thus, the algorithm provides a minimal feature subset. So, **you do not have to decide on how many features to take. PyImpetus selects the optimal set for you.**

PyImpetus has been completely revamped and now supports **binary classification, multi-class classification and regression** tasks. It has been tested on 14 datasets and outperformed state-of-the-art Markov Blanket learning algorithms on all of them along with traditional feature selection algorithms such as Forward Feature Selection, Backward Feature Elimination and Recursive Feature Elimination.

## How to install?
```pip install PyImpetus```

## Functions and parameters
```python
# The initialization of PyImpetus takes in multiple parameters as input
# PPIMBC is for classification
model = PPIMBC(model, p_val_thresh, num_simul, simul_size, simul_type, sig_test_type, cv, verbose, random_state, n_jobs)
```
- **model** - `estimator object, default=DecisionTreeClassifier()` The model which is used to perform classification in order to find feature importance via significance-test. 
- **p_val_thresh** - `float, default=0.05` The p-value (in this case, feature importance) below which a feature will be considered as a candidate for the final MB.
- **num_simul** - `int, default=30` **(This feature has huge impact on speed)** Number of train-test splits to perform to check usefulness of each feature. For large datasets, the value should be considerably reduced though do not go below 5.
- **simul_size** - `float, default=0.2` The size of the test set in each train-test split
- **simul_type** - `boolean, default=0` To apply stratification or not
	- `0` means train-test splits are not stratified.
	- `1` means the train-test splits will be stratified.
- **sig_test_type** - `string, default="non-parametric"` This determines the type of significance test to use.
	- `"parametric"` means a parametric significance test will be used (Note: This test selects very few features)
	- `"non-parametric"` means a non-parametric significance test will be used
- **cv** - `cv object/int, default=0` Determines the number of splits for cross-validation. Sklearn CV object can also be passed. A value of 0 means CV is disabled.
- **verbose** - `int, default=2` Controls the verbosity: the higher, more the messages.
- **random_state** - `int or RandomState instance, default=None` Pass an int for reproducible output across multiple function calls.
- **n_jobs** - `int, default=-1` The number of CPUs to use to do the computation.
	- `None` means 1 unless in a `:obj:joblib.parallel_backend` context.
	- `-1` means using all processors.

```python
# The initialization of PyImpetus takes in multiple parameters as input
# PPIMBR is for regression
model = PPIMBR(model, p_val_thresh, num_simul, simul_size, sig_test_type, cv, verbose, random_state, n_jobs)
```
- **model** - `estimator object, default=DecisionTreeRegressor()` The model which is used to perform regression in order to find feature importance via significance-test. 
- **p_val_thresh** - `float, default=0.05` The p-value (in this case, feature importance) below which a feature will be considered as a candidate for the final MB.
- **num_simul** - `int, default=30` **(This feature has huge impact on speed)** Number of train-test splits to perform to check usefulness of each feature. For large datasets, the value should be considerably reduced though do not go below 5.
- **simul_size** - `float, default=0.2` The size of the test set in each train-test split
- **sig_test_type** - `string, default="non-parametric"` This determines the type of significance test to use.
	- `"parametric"` means a parametric significance test will be used (Note: This test selects very few features)
	- `"non-parametric"` means a non-parametric significance test will be used
- **cv** - `cv object/int, default=0` Determines the number of splits for cross-validation. Sklearn CV object can also be passed. A value of 0 means CV is disabled.
- **verbose** - `int, default=2` Controls the verbosity: the higher, more the messages.
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
from PyImpetus import PPIMBC, PPIMBR
```

## Usage
```python
# Import the algorithm. PPIMBC is for classification and PPIMBR is for regression
from PyImeptus import PPIMBC, PPIMBR
# Initialize the PyImpetus object
model = PPIMBC(model=SVC(random_state=27, class_weight="balanced"), p_val_thresh=0.05, num_simul=30, simul_size=0.2, simul_type=0, sig_test_type="non-parametric", cv=5, random_state=27, n_jobs=-1, verbose=2)
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
Note: Play with the values of **num_simul**, **simul_size**, **simul_type** and **p_val_thresh** because sometimes a specific combination of these values will end up giving best results
- ~~Increase the **cv** value~~ In all experiments, **cv** did not help in getting better accuracy. Use this only when you have extremely small dataset
- Increase the **num_simul** value
- Try one of these values for **simul_size** = `{0.1, 0.2, 0.3, 0.4}`
- Use non-linear models for feature selection. Apply hyper-parameter tuning on models
- Increase value of **p_val_thresh** in order to increase the number of features to include in thre Markov Blanket

## For better speeds
- ~~Decrease the **cv** value. For large datasets cv might not be required. Therefore, set **cv=0** to disable the aggregation step. This will result in less robust feature subset selection but at much faster speeds~~
- Decrease the **num_simul** value but don't decrease it below 5
- Set **n_jobs** to -1
- Use linear models

## For selection of less features
- Try reducing the **p_val_thresh** value
- Try out `sig_test_type = "parametric"`

## Performance in terms of Accuracy (classification) and MSE (regression)
| Dataset | # of samples | # of features | Task Type | Score using all features | Score using [featurewiz](https://github.com/AutoViML/featurewiz) | Score using PyImpetus | # of features selected | % of features selected | Tutorial |
| --- | --- | --- | --- |--- |--- |--- |--- |--- |--- |
| Ionosphere | 351 | 34 | Classification | 88.01% |  | 92.86% | 14 | 42.42% | [tutorial here](https://github.com/atif-hassan/PyImpetus/blob/master/tutorials/Classification_Tutorial.ipynb) |
| Arcene | 100 | 10000 | Classification | 82% |  | 84.72% | 304 | 3.04% | |
| AlonDS2000 | 62 | 2000 | Classification | 80.55% | 86.98% | 88.49% | 75 | 3.75% | |
| slice_localization_data | 53500 | 384 | Regression | 6.54 |  | 5.69 | 259 | 67.45% | [tutorial here](https://github.com/atif-hassan/PyImpetus/blob/master/tutorials/Regression_Tutorial.ipynb) |

**Note**: Here, for the first, second and third tasks, a higher accuracy score is better while for the fourth task, a lower MSE (Mean Squared Error) is better.

## Performance in terms of Time (in seconds)
| Dataset | # of samples | # of features | Time (with PyImpetus) |
| --- | --- | --- | --- |
| Ionosphere | 351 | 34 | 35.37 |
| Arcene | 100 | 10000 | 1570 |
| AlonDS2000 | 62 | 2000 | 125.511 |
| slice_localization_data | 53500 | 384 | 1296.13 |

## Future Ideas
- Let me know

## Feature Request
Drop me an email at **atif.hit.hassan@gmail.com** if you want any particular feature

# Please cite this work as
@misc{hassan2021ppfs,
      title={PPFS: Predictive Permutation Feature Selection}, 
      author={Atif Hassan and Jiaul H. Paik and Swanand Khare and Syed Asif Hassan},
      year={2021},
      eprint={2110.10713},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
