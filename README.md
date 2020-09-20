# PyImpetus
PyImpetus is a feature selection algorithm that picks features by considering their performance both individually as well as conditioned on other selected features. This allows the algorithm to not only select the best set of features, it also selects the best set of features that play well with each other. For example, the best performing feature might not play well with others while the remaining features, when taken together could out-perform the best feature. PyImpetus takes this into account and produces the best possible combination.

PyImpetus is basically the interIAMB algorithm as provided in the paper, titled, (An Improved IAMB Algorithm for Markov Blanket Discovery)[http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.348.4667&rep=rep1&type=pdf#page=137] [1], with the conditional mutual information part being replaced by a conditional test. This test is as described in the paper, titled, (Testing Conditional Independence in Supervised Learning Algorithms)[https://arxiv.org/abs/1901.09917] [2].

## References
<a id="1">[1]</a> 
Zhang, Y., Zhang, Z., Liu, K., & Qian, G. (2010).
An Improved IAMB Algorithm for Markov Blanket Discovery.
JCP, 5(11), 1755-1761.

<a id="2">[2]</a>
Watson, D. S., & Wright, M. N. (2019).
Testing Conditional Independence in Supervised Learning Algorithms.
arXiv preprint arXiv:1901.09917.
