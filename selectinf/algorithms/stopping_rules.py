"""
Stopping rules used in sequential FDR control.

See `http://arxiv.org/abs/1309.5352`_

"""

import numpy as np

def simple_stop(pvalues, alpha):
    """
    Compute the number of rejections using 
    simple stop, the first time a p-value is above
    alpha.

    Parameters
    ----------

    pvalues : np.float

    alpha : float

    Returns
    -------

    num_rejections : int

    """
    if not np.all(pvalues <= alpha):
        return np.min(np.nonzero(pvalues > alpha)[0])
    else:
        return pvalues.shape[0]

def strong_stop(pvalues, alpha):
    """

    Compute the number of rejections using 
    strong stop of `http://arxiv.org/abs/1309.5352`_

    >>> strong_stop(np.array([0.5,0.6,0.7,0.8,0.9]), 0.05)
    0
    >>> strong_stop(np.array([0.001, 0.002, 0.0015, 0.0013, 0.05, 0.6]), 0.05)
    3

    In R:

    > strongstop(c(0.001, 0.002, 0.0015, 0.0013, 0.05, 0.6), 0.05)
    [1] 3
    > strongstop(c(0.5,0.6,0.7,0.8,0.9), 0.05)
    [1] 0

    Parameters
    ----------

    pvalues : np.float

    alpha : float

    Returns
    -------

    num_rejections : int

    Based on R code:
    ----------------

    strongstop <- function(p.values,alpha) {
       d <- length(p.values)
       lhs <- exp(rev(cumsum(rev(log(p.values)/(1:d))))) # LHS from G'Sell et al.
       rhs <- alpha * (1:d) / d # RHS from G'Sell et al.
       return(max(c(0,which(lhs <= rhs))))
    }

    """
    n = pvalues.shape[0]
    LHS = np.exp(np.cumsum((np.log(pvalues) / np.linspace(1., n, n))[::-1])[::-1])
    RHS = alpha * np.linspace(1., n, n) / n
    if np.any(LHS <= RHS):
        return max(np.nonzero(LHS <= RHS)[0])+1
    return 0


def forward_stop(pvalues, alpha):
    """

    Compute the number of rejections using 
    forward stop of  `http://arxiv.org/abs/1309.5352`_

    >>> forward_stop(np.array([0.5,0.6,0.7,0.8,0.9]), 0.05)
    0
    >>> forward_stop(np.array([0.001, 0.002, 0.0015, 0.0013, 0.05, 0.6]), 0.05)
    5

    In R:

    > forwardstop(c(0.5,0.6,0.7,0.8,0.9), 0.05)
    [1] 0
    > forwardstop(c(0.001, 0.002, 0.0015, 0.0013, 0.05, 0.6), 0.05)
    [1] 5
    > 

    Parameters
    ----------

    pvalues : np.float

    alpha : float

    Returns
    -------

    num_rejections : int

    Based on R code:
    ----------------

    forwardstop <- function(p, alpha) {
       m <- length(p)
       sums <- -(1/(1:m))*cumsum(log(1-p))
       return(max(c(0, which(sums < alpha))))
    }

    """

    n = pvalues.shape[0]
    sums = (-1. / np.linspace(1, n, n)) * np.cumsum(np.log(1 - pvalues))
    if np.any(sums < alpha):
        return max(np.nonzero(sums < alpha)[0])+1
    return 0


