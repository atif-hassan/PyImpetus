[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/PyImpetus.svg)](https://pypi.python.org/pypi/PyImpetus/)
[![Downloads](https://pepy.tech/badge/PyImpetus)](https://pepy.tech/project/PyImpetus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/PyImpetus/commits/master)
# PyImpetus
PyImpetus is a **feature selection algorithm** that picks features by considering their performance both individually as well as conditioned on other selected features. This allows the algorithm to not only select the best set of features, but also select the **best set of features that play well with each other**. For example, the best performing feature might not play well with others while the remaining features, when taken together could out-perform the best feature. PyImpetus takes this into account and produces the best possible combination.

PyImpetus has been completely revamped and borrows ideas from multiple papers, most significantly from [Testing Conditional Independence in Supervised Learning Algorithms](https://arxiv.org/abs/1901.09917) [1]. It uses a novel CV based aggregation method to recommend the most roubst set of minimal features (Markov Blanket).

PyImpetus was tested on 13 datasets and outperformed state-of-the-art Markov Blanket learning algorithms on all of them.

## How to install?
```pip install PyImpetus```

## Functions and parameters
```python
# The initialization of PyImpetus takes in multiple parameters as input
model = CPIMB(model, p_val_thresh, num_simul, cv, verbose, random_state, n_jobs)
```
- **model** - `estimator object, default=DecisionTreeClassifier()` The model which is used to perform classification/regression in order to find feature importance via t-test. The idea is that, you don't want to use a linear model as you won't be able to pick any non-linear relationship that a single feature has with other features or the target variable. For non-linear models, one should use heavily regularized complex models or a simple decision tree which requires little to no pre-processing. Therefore, the default model is a decision tree.
- **p_val_thresh** - `float, default=0.05` The p-value (in this case, feature importance) below which a feature will be considered as a candidate for the final MB.
- **num_simul** - `int, default=30` **(This feature has huge impact on speed)** Number of train-test splits to perform to check usefulness of each feature. For large datasets, this size should be considerably reduced though do not go below 5.
- **cv** - `int, default=5` Determines the the number of splits for cross-validation.	
- **verbose** - `int, default=0` Controls the verbosity: the higher, more the messages.
- **random_state** - `int or RandomState instance, default=None` Pass an int for reproducible output across multiple function calls.
- **n_jobs** - `int, default=-1` The number of CPUs to use to do the computation.
	- `None` means 1 unless in a `:obj:joblib.parallel_backend` context.
	- `-1` means using all processors.

```python
# To fit PyImpetus on provided dataset
fit(data, target)
```
- **data** - A pandas dataframe upon which feature selection is to be applied
- **target** - A numpy array, denoting the target variable

```python
# This function returns the names of the columns that form the MB (These are the recommended features)
transform()
```

```python
# To fit PyImpetus on provided dataset and return recommended features
fit_transform(data, target)
```
- **data** - A pandas dataframe upon which feature selection is to be applied
- **target** - A numpy array, denoting the target variable

## How to import?
```python
from CPIMBC import CPIMB
```

## Usage
```python
# Import the algorithm
from CPIMBC import CPIMB
# Initialize the PyImpetus object
model = CPIMB(model=SVC(random_state=27, class_weight="balanced"), p_val_thresh=0.05, num_simul=30, cv=5, random_state=27, n_jobs=-1, verbose=2)
# The fit_transform function runs the algorithm, finds the best set of features and returns the recommended features
MB = model.fit_transform(df.drop("Response", axis=1), df["Response"].values)
# Now create your pruned dataset
df = df[MB]
```

## Timeit!
On a dataset of **381,110** samples and **10** features, PyImpetus took 77.6 seconds to find the best set of minimal features. This is in contrast with the previous version of PyImpetus which took 609 seconds for the same dataset. This test was performed on a 10th gen corei7 with n_jobs set tot -1.

## Tutorials
You can find a usage [tutorial here](https://github.com/atif-hassan/PyImpetus/blob/master/tutorials/Tutorial.ipynb).

## Future Ideas
- The conditional test might change in the near future

## Feature Request
Drop me an email at **atif.hit.hassan@gmail.com** if you want any particular feature

## References
<a id="1">[1]</a>
Watson, D. S., & Wright, M. N. (2019).
Testing Conditional Independence in Supervised Learning Algorithms.
arXiv preprint arXiv:1901.09917.
