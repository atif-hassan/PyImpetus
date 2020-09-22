[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/PyImpetus.svg)](https://pypi.python.org/pypi/PyImpetus/)
[![Downloads](https://pepy.tech/badge/PyImpetus)](https://pepy.tech/project/PyImpetus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/PyImpetus/commits/master)
# PyImpetus
PyImpetus is a **feature selection algorithm** that picks features by considering their performance both individually as well as conditioned on other selected features. This allows the algorithm to not only select the best set of features, it also selects the **best set of features that play well with each other**. For example, the best performing feature might not play well with others while the remaining features, when taken together could out-perform the best feature. PyImpetus takes this into account and produces the best possible combination.

PyImpetus is basically the **interIAMB** algorithm as provided in the paper, titled, [An Improved IAMB Algorithm for Markov Blanket Discovery](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.348.4667&rep=rep1&type=pdf#page=137) [1] with the conditional mutual information part being replaced by a **conditional test**. This test is as described in the paper, titled, [Testing Conditional Independence in Supervised Learning Algorithms](https://arxiv.org/abs/1901.09917) [2].

## How to install?
```pip install PyImpetus```

## Functions and parameters
```python
# The initialization of PyImpetus takes in multiple parameters as input
fs = inter_IAMB(model, min_feat_proba_thresh, p_val_thresh, k_feats_select, num_simul, cv, regression, verbose, random_state, n_jobs, pre_dispatch)
```
- **model** - `estimator object, default=None` The model which is used to perform classification/regression in order to find feature importance via t-test. The idea is that, you don't want to use a linear model as you won't be able to pick any non-linear relationship that a single feature has with other features or the target variable. For non-linear models, one should use heavily regularized complex models or a simple decision tree which requires little to no pre-processing. Therefore, the default model is a decision tree.
- **min_feat_proba_thresh** - `float, default=0.1` The minimum probability of occurrence that a feature should possess over all folds for it to be considered in the final Markov Blanket (MB) of target variable
- **p_val_thresh** - `float, default=0.05` The p-value (in this case, feature importance) below which a feature will be considered as a candidate for the final MB.
- **k_feats_select** - `int, default=5` The number of features to select during growth phase of InterIAMB algorithm. Larger values give faster results. Effect of large values has not yet been tested.
- **num_simul** - `int, default=100` Number of train-test splits to perform to check usefulness of each feature. For large datasets, this size should be considerably reduced though do not go below 10.
- **cv** - `int, cross-validation generator or an iterable, default=None` Determines the cross-validation splitting strategy.
	Possible inputs for cv are:

	- None, to use the default 5-fold cross validation,
	- integer, to specify the number of folds in a (Stratified)KFold,
	- CV splitter,
	- An iterable yielding (train, test) splits as arrays of indices.

	For integer/None inputs, if the `regression` param is False and `y` is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
- **regression** - `bool, default=False` Defines the task - whether it is regression or classification.
- **verbose** - `int, default=0` Controls the verbosity: the higher, more the messages.
- **random_state** - `int or RandomState instance, default=None` Pass an int for reproducible output across multiple function calls.
- **n_jobs** - `int, default=None` The number of CPUs to use to do the computation.
	- `None` means 1 unless in a `:obj:joblib.parallel_backend` context.
	- `-1` means using all processors.
- **pre_dispatch** - `int or str, default='2*n_jobs'` Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. This parameter can be:
	- None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
	- An int, giving the exact number of total jobs that are spawned
	- A str, giving an expression as a function of n_jobs, as in `2*n_jobs`

```python
# This function returns a list of features for input pandas dataset
fit(data, target)
```
- **data** - A pandas dataframe upon which feature selection is to be applied
- **target** - A string denoting the target variable in the dataframe.

```python
# This function returns a pruned dataframe consisting of only the selected features
transform(data)
```
- **data** - The dataframe which is to be pruned to the selected features

## Attributes
- **final_feats_** - `ndarray or list of ndarray of shape (n_classes,)` Final list of features.

## How to import?
```python
from PyImpetus import inter_IAMB
```

## Usage
```python
# Initialize the PyImpetus object
fs = inter_IAMB(num_simul=10, n_jobs=-1, verbose=2, random_state=27)
# The fit function runs the algorithm and finds the best set of features
fs.fit(df_train_.drop("Response", axis=1), df_train_["Response"])
# You can check out the selected features using the "final_feats_" attribute
feats = fs.final_feats_
# The transform function prunes your pandas dataset to the set of final features
X_train = fs.transform(df_train).values
# Prune the test dataset as well
X_test = fs.transform(df_test).values
```

## Timeit!
On a dataset of **381,110** samples and **10** features, PyImpetus took approximately **1.68** minutes on each fold of a 5-fold CV with the final set of features being selected at around **8.4** minutes. This test was performed on a 10th gen corei7 with n_jobs set tot -1.

## Tutorials
You can find a usage [tutorial here](https://github.com/atif-hassan/PyImpetus/blob/master/tutorials/Tutorial.ipynb). I got a huge boost in AnalyticVidhya's JanataHack: Cross-sell Prediction hackathon. I jumped from rank **223/600 to 166/600 just by using the features recommended by PyImpetus**. I was also able to **out-perform SOTA in terms of f1-score by about 4% on Alzheimer disease dataset using PyImpetus**. The paper is currently being written.

## Future Ideas
- ~~Multi-threading CV in order to drastically reduce computation~~ (DONE thanks to [Antoni Baum](https://github.com/Yard1))

## Feature Request
Drop me an email at **atif.hit.hassan@gmail.com** if you want any particular feature

## Special Shout Out
Thanks to [Antoni Baum](https://github.com/Yard1) who restructured my code to match the sklearn API, added parallelism and extensive documentation in code along with other miscellaneous stuff. Seriously man, thank you!!

## References
<a id="1">[1]</a> 
Zhang, Y., Zhang, Z., Liu, K., & Qian, G. (2010).
An Improved IAMB Algorithm for Markov Blanket Discovery.
JCP, 5(11), 1755-1761.

<a id="2">[2]</a>
Watson, D. S., & Wright, M. N. (2019).
Testing Conditional Independence in Supervised Learning Algorithms.
arXiv preprint arXiv:1901.09917.
