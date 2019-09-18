"""
Step 1 test based on largest singular vector. 

This is the test described in `Kac Rice`_ for $X=I$ and the penalty being the nuclear norm

.. math::

     {\cal P}(\beta) = \sim_{i=1}^{\text{min(n,p)}} \sigma_i(\beta)

for $\beta \in \mathbb{R}^{n \times p}$.

.. _Kac Rice: http://arxiv.org/abs/1308.3020
"""

import numpy as np
from ..distributions.pvalue import general_pvalue

def pvalue(X, sigma=1, nsim=5000):
    n, p = X.shape
    D = np.linalg.svd(X)[1] / sigma
    m = n+p-2
    H = np.zeros(m)
    
    nonzero = np.hstack([D[1:],-D[1:]])
    H[:nonzero.shape[0]] = nonzero
        
    return max(0, min(general_pvalue(D[0], D[1], np.inf, H, nsim=nsim), 1))
