[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/PyImpetus.svg)](https://pypi.python.org/pypi/PyImpetus/)
[![Downloads](https://pepy.tech/badge/reg-resampler)](https://pepy.tech/project/PyImpetus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/PyImpetus/commits/master)
# PyImpetus
PyImpetus is a feature selection algorithm that picks features by considering their performance both individually as well as conditioned on other selected features. This allows the algorithm to not only select the best set of features, it also selects the best set of features that play well with each other. For example, the best performing feature might not play well with others while the remaining features, when taken together could out-perform the best feature. PyImpetus takes this into account and produces the best possible combination.

PyImpetus is basically the interIAMB algorithm as provided in the paper, titled, [An Improved IAMB Algorithm for Markov Blanket Discovery](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.348.4667&rep=rep1&type=pdf#page=137) [1] with the conditional mutual information part being replaced by a conditional test. This test is as described in the paper, titled, [Testing Conditional Independence in Supervised Learning Algorithms](https://arxiv.org/abs/1901.09917) [2].

## How to install?
```pip install reg_resampler```

## Functions and parameters
```python
# This returns a numpy list of classes for each corresponding sample. It also automatically merges classes when required
fit(X, target, bins=3, min_n_samples=6, balanced_binning=False, verbose=2)
```
- **X** - Either a pandas dataframe or numpy matrix. Complete data to be resampled.
- **target** - Either string (for pandas) or index (for numpy). The target variable to be resampled.
- **bins=3** - The number of classes that the user wants to generate. (Default: 3)
- **min_n_samples=6** - Minimum number of samples in each bin. Bins having less than this value will be merged with the closest bin. Has to be more than neighbours in imblearn. (Default: 6)
- **balanced_binning=False** - Decides whether samples are to be distributed roughly equally across all classes. (Default: False)
- **verbose=2** - 0 will disable print by package, 1 will print info about class mergers and 2 will also print class distributions.

```python
# Performs resampling and returns the resampled dataframe/numpy matrices in the form of data and target variable.
resample(sampler_obj, trainX, trainY)
```
- **sampler_obj** - Your favourite resampling algorithm's object (currently supports imblearn)
- **trainX** - Either a pandas dataframe or numpy matrix. Data to be resampled. Also, contains the target variable
- **trainY** - Numpy array of psuedo classes obtained from fit function.

### Important Note
All functions return the same data type as provided in input.

## How to import?
```python
from reg_resampler import resampler
```

## Usage
```python
# Initialize the resampler object
rs = resampler()

# You might recieve info about class merger for low sample classes
# Generate classes
Y_classes = rs.fit(df_train, target=target, bins=num_bins)
# Create the actual target variable
Y = df_train[target]

# Create a smote (over-sampling) object from imblearn
smote = SMOTE(random_state=27)

# Now resample
final_X, final_Y = rs.resample(smote, df_train, Y_classes)
```

## Tutorials
You can find a usage [tutorial here](https://github.com/atif-hassan/Regression_ReSampling/tree/master/tutorials). I got a huge boost in the JanataHack

## Future Ideas
- Multi-threading the CV in order to drastically reduce the computation time

## Feature Request
Drop me an email at **atif.hit.hassan@gmail.com** if you want any particular feature

## References
<a id="1">[1]</a> 
Zhang, Y., Zhang, Z., Liu, K., & Qian, G. (2010).
An Improved IAMB Algorithm for Markov Blanket Discovery.
JCP, 5(11), 1755-1761.

<a id="2">[2]</a>
Watson, D. S., & Wright, M. N. (2019).
Testing Conditional Independence in Supervised Learning Algorithms.
arXiv preprint arXiv:1901.09917.
