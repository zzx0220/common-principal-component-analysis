# Common Principal Component Analysis (CPCA)

## Introduction
Common Principal Component Analysis (CPCA) is a generalized form of Principal Component Analysis (PCA) aimed at identifying common principal components of several distinct datasets. When it is assumed that the datasets collected under different conditions have the same components but different weights, CPCA can be used to identify these common components and their weights. The algorithm is based on eigenvalue decomposition.

Similar approach can be extended to factor analysis to find common common-factors of several datasets. The covariance/correlation matrices of the datasets can be decomposed into two parts (Equation 5.13, section 5.3.2, ref.):

$$
R_{XX}=QD^2Q^T+\Psi^2
$$

where the first term corresponds to the common factors, and the second term corresponds to the unique factors. For multiple datasets, assume $Q$ is consistent across all datasets.



## Main Files
- `common_eig.m`: Computes the common eigenvectors and eigenvalues of two covariance matrices using LS definition (9.6 in ref.). 
- `cpca.m`: Implements Common Principal Component Analysis to find common principal components of two datasets.
- `fa_across_conds.m`: Implements factor analysis across multiple conditions to find common common-factors.

## Installation and Dependencies
- Ensure that your MATLAB environment has the Manopt toolbox installed.
- You can download the Manopt toolbox from the [Manopt website](http://www.manopt.org/) and follow the installation instructions provided there.

## Usage

```matlab
eigen_result = common_eig(covmat, options);
all_result = cpca(data1, data2);
result = fa_across_conds(corrmats, options);
```

## References
Trendafilov, N., & Gallo, M. (2021). Multivariate Data Analysis on Matrix Manifolds: (With Manopt). Springer International Publishing. https://doi.org/10.1007/978-3-030-76974-1
