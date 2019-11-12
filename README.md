# Weighted Non-Negative Matrix Factorization

## About
`wNMF` implements a simple version of Non-Negative Matrix Factorization (NMF) that utilizes a weight matrix to weight the importance of each feature in each sample of the data matrix to be factorized. It mimics the `sklearn` model API, as well as allows access to results from multiple optimization runs.  

More information about the modified multiplicative update algorithim utilized can be found here:
[Blondel, Vincent & Ho, Ngoc-Diep & Van Dooren, Paul. (2007). Weighted Nonnegative Matrix Factorization and Face Feature Extraction](https://pdfs.semanticscholar.org/e20e/98642009f13686a540c193fdbce2d509c3b8.pdf) 

`wNMF` specifically implements solutions for determining the decomposed matrices U and V when minimizing the Frobenius Norm or the Kullback-Leibler Divergence:

**Useful Links**
- [Source on Github](https://github.com/asn32/weighted-nmf)
- [Package on PyPI](https://pypi.org/project/weighted-nmf/)

## Installation
This package is available on PyPI and can be installed with `pip`:
```bash
$ pip install wNMF
```

Alternatively, download the source from [github](https://github.com/asn32/weighted-nmf) and install:
```bash
$ git clone https://github.com/asn32/weighted-nmf
$ cd weighted-nmf
$ pip install .
```

## Usage
`wNMF` is a python library that can be imported.
```python
import wNMF
```
And the main functionality is implemented to mimic using an `sklearn.decomposition` model. 

First create an instance of the `wNMF` model setting the number of components, then fit the model to the data using the instance methods `wNMF().fit` or `wNMF().fit_transform`.
Parameters such as the maximum number of iterations to perform `max_iter` or whether to track individual errors in runs `track_error` can also be set.
```python
import numpy as np
from wNMF import wNMF

## Mock data, a 100x100 data matrix, reduce to 25 dimensions
n=100
features = 100
components=25
X = 100*np.random.uniform(size=n*features).reshape(features,n)
W = np.ones_like(X)

## Define the model / fit
model = wNMF(n_components=25,
            beta_loss='kullback-leibler',
            max_iter=1000,
            track_error=True)

fit = model.fit(X=X,W=W,n_run=5)
```

After the fit is complete, explore the fit quality by examining the decomposed matrices and / or overall error. 
```python
print(fit.err)
print(fit.V)
print(fit.U)
print(fit.err_all)
```

## License
wnmf is MIT-licensed

## Disclaimer
`wnmf` is provided with no guarantees

