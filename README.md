[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/atif-hassan/)

[![PyPI version shields.io](https://img.shields.io/pypi/v/PyImpetus.svg)](https://pypi.python.org/pypi/PyImpetus/)
[![Downloads](https://pepy.tech/badge/reg-resampler)](https://pepy.tech/project/PyImpetus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/atif-hassan/PyImpetus/commits/master)
# PyImpetus
PyImpetus is a feature selection algorithm that picks features by considering their performance both individually as well as conditioned on other selected features. This allows the algorithm to not only select the best set of features, it also selects the best set of features that play well with each other. For example, the best performing feature might not play well with others while the remaining features, when taken together could out-perform the best feature. PyImpetus takes this into account and produces the best possible combination.

PyImpetus is basically the interIAMB algorithm as provided in the paper, titled, [An Improved IAMB Algorithm for Markov Blanket Discovery](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.348.4667&rep=rep1&type=pdf#page=137) [1] with the conditional mutual information part being replaced by a conditional test. This test is as described in the paper, titled, [Testing Conditional Independence in Supervised Learning Algorithms](https://arxiv.org/abs/1901.09917) [2].

## How to install?
```pip install PyImpetus```

## Functions and parameters
```python
# The initialization of PyImpetus takes in multiple parameters as input
fs = inter_IAMB(model=None, min_feat_proba_thresh=0.1, p_val_thresh=0.05, k_feats_select=5, num_simul=100, stratified=False, num_cv_splits=5, regression=False, verbose=1)
```
- **model** - The model which is used to find perform classification/regression in order to find feature importance via t-test. By default, a decision tree is used. The idea is that, you don't want to use a linear model as you won't be able to pick any non-linear relationship that a single feature has with other features or the target variable. For non-linear models, one should use heavily regularized complex models or a simple decision tree which requires little to no pre-processing. Therefore, the default model is a decision tree
- **target** - A string denoting the target variable in the dataframe.

```
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

## How to import?
```python
from PyImpetus import inter_IAMB
```

## Usage
```python
# Initialize the resampler object
fs = inter_IAMB(num_simul=10)
# The fit function returns a list of the features selected
feats = fs.fit(df_train_, "Response")
# The transform function prunes your pandas dataset to the set of final features
X_train = fs.transform(df_train).values
# Prune the test dataset as well
X_test = fs.transform(df_test).values

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
You can find a usage [tutorial here](https://github.com/atif-hassan/PyImpetus/blob/master/tutorials/Tutorial.ipynb). I got a huge boost in AnalyticVidhya's JanataHack: Cross-sell Prediction hackathon. I jumped from rank 223/600 to 166/600 just by using the features recommended by PyImpetus. I was also able to out-perform SOTA in terms of f1-score by about 4% on Alzheimer disease dataset using PyImpetus. The paper is currently being written.

## Future Ideas
- Multi-threading CV in order to drastically reduce computation time

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
