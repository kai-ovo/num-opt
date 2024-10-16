# Numerical Optimization
This repository contains implementations of numerous popular numerical algorithms, including proximal algorithms and their popular variants, and Newton's methods and their variants. Two case studies in the form of Jupyter Notebooks are also provided: the LASSO problem and the Non-negative Matrix Factorization (NMF) problem. In particular for the NMF problem, we conduct extensive literature review and mathematically prove some insightful algorithmic properties.

---

 `proximal.py` contains the following algorithms:
- Proximal Point Algorithm (PPA)
- Proximal Gradient Method (PGM)
- Accelerated Proximal Gradient Method (APGM)
- Alternating Direction Method of Multipliers (ADMM)
- Forward-Backward Splitting (FBS)
- Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) 
- Linearized ADMM
- Douglas-Rachford Splitting method

`newton.py` contains the following algorithms:
- Basic Newton's method for unconstrained optimization
- Modified Newton's Method for optimization with Hessian modification
- BFGS
- L-BFGS
- Davidon-Fletcher-Powell method
- Inexact Newton's Method with CG
- Gauss-Newton

`nmf.ipynb` is the notebook for the Non-negative Matrix Factorization problem solved with different algorithms

`lasso.ipynb` is the notebook for the LASSO problem solved with different algorithms

`test_newton.py` and `test_prox.py` contain testing scripts for various Newton's methods and proximal methods respectively

## References

- Parikh, Neal, and Stephen Boyd. "Proximal algorithms." Foundations and trends in Optimization 1.3 (2014): 127-239.
- Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends in Machine learning 3.1 (2011): 1-122.
- Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization." (2006).
- Bottou, Léon, Frank E. Curtis, and Jorge Nocedal. "Optimization methods for large-scale machine learning." SIAM review 60.2 (2018): 223-311.
- Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM journal on imaging sciences 2.1 (2009): 183-202.
