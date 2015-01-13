"""
In this module, we implement forward stepwise model selection for $K$ steps.

The main goal of this is to produce a set of linear inequality constraints satisfied by
$y$ after $K$ steps.

"""

import numpy as np
import scipy.sparse
from scipy.stats import norm as ndist

# local imports 

from .affine import constraints
from .chisq import quadratic_test

DEBUG = False

class projection(object):

    """
    A projection matrix, U an orthonormal basis of the column space.
    
    Warning: we do not check if U has orthonormal columns. 

    This can be enforced by calling the `orthogonalize` method
    which returns a new instance.
    
    """
    def __init__(self, U):
        self.U = U

    def __call__(self, Z, rank=None):
        if rank is None:
            return np.dot(self.U, np.dot(self.U.T, Z))
        else:
            return np.dot(self.U[:,:rank], np.dot(self.U.T[:rank], Z))

    def stack(self, Unew):
        """
        Form a new projection matrix by hstack.

        Warning: no check is mode to ensure U has orthonormal columns.
        """
        return projection(np.hstack([self.U, Unew]))

    def orthogonalize(self):
        """
        Force columns to be orthonormal.
        """
        return projection(np.linalg.svd(self.U, full_matrices=False)[0])

class forward_stepwise(object):

    """
    Centers columns of X!
    """

    def __init__(self, X, Y, 
                 subset=None,
                 covariance=None):
        if subset is None:
            subset = np.ones(Y.shape[0], np.bool)
        self.X = X.copy()[subset]
        self.X -= self.X.mean(0)[None,:]
        self.Y = Y.copy()[subset]
        self.Y -= self.Y.mean()
        self.P = [None] # residual forming projections
        self.A = None
        self.variables = []
        self.Z = []
        self.signs = []
        if covariance is None:
            covariance = np.identity(self.X.shape[0])
        self.covariance = covariance

    def __iter__(self):
        return self

    def next(self):
        """
        Take one step of forward stepwise.
        Internally, this has the effect of: 

        * adding a new (lowrank) projection to `self.P`, 
        
        * adding a new variable to `self.variables`

        * adding a certain number of rows to `self.A`

        * signs are also tracked (unnecessarily for the moment) in `self.signs`

        The multiplication `np.dot(self.A, eta)` can be made more 
        efficient because the projections are just a list of 
        Gram-Schmidt orthogonalized vectors.

        """
        P = self.P[-1]
        
        X, Y = self.X, self.Y
        n, p = self.X.shape

        if P is None: # first step
            U = np.dot(X.T, Y)
            scale = np.sqrt((X**2).sum(0))
            Z = np.fabs(U) / scale
            idx = np.argmax(Z)
            sign = np.sign(U[idx])
            Unew = X[:,idx] / scale[idx]
            Pnew = projection(Unew.reshape((-1,1)))
            self.As = [canonicalA(X, Y, idx, sign, scale=scale)]
            self.A = self.As[0]
            self.variables.append(idx)
            self.signs.append(sign)
            self.Z.append(Z[idx])
        else:
            RY = Y-P(Y)
            RX = X-P(X)
            keep = np.ones(p, np.bool)
            keep[self.variables] = 0
            RX = RX[:,keep]

            scale = np.sqrt((RX**2).sum(0))
            U = np.dot(RX.T, RY)
            Z = np.fabs(U) / scale
            idx = np.argmax(Z)

            sign = np.sign(U[idx])
            self.variables.append(np.arange(p)[keep][idx])
            self.signs.append(sign)
            self.Z.append(Z[idx])

            Unew = RX[:,idx] / scale[idx]
            Pnew = P.stack(Unew.reshape((-1,1)))
            newA = canonicalA(RX, RY, idx, sign, scale=scale)
            self.As.append(newA)
            if DEBUG:
                print np.linalg.norm(np.dot(newA, Y) - np.dot(newA, RY)), 'should be 0'
                print np.linalg.norm(P(newA.T)), np.linalg.norm(P(RX)), 'newA'
            self.A = np.vstack([self.A, newA])

        if DEBUG:
            Pother = np.linalg.svd(X[:,self.variables], full_matrices=0)[0]
            print np.linalg.norm(Pother - Pnew(Pother)), 'Pnorm'
            print self.variables, 'selected variables'
            print self.signs, 'signs'
            print self.A.shape, 'A shape'
            print np.dot(self.A, Y).max(), 'should be nonpositive'

        self.P.append(Pnew)

    @property
    def constraints(self):
        return constraints(self.A, np.zeros(self.A.shape[0]), 
                           covariance=self.covariance)

    def model_intervals(self, which_step, alpha=0.05, UMAU=False):
        """
        Compute selection intervals for
        a given step of forward stepwise.

        Parameters
        ----------

        which_step : int
            Which step of forward stepwise.

        sigma : float
            Standard deviation of noise.

        alpha : float
            1 - confidence level for intervals.

        UMAU : bool
            Use UMAU intervals or equal-tailed intervals?

        Returns
        -------

        intervals : list
             List of (variable, LS_direction, LS_estimate, interval)
             where LS_direction is the vector that computes this variables
             least square coefficient in the current model, and LS_estimate
             is sum(LS_estimate * self.Y).

        """
        C = self.constraints
        intervals = []
        LSfunc = np.linalg.pinv(self.X[:,self.variables[:which_step]])
        for i in range(LSfunc.shape[0]):
            eta = LSfunc[i]
            _interval = C.interval(eta, self.Y,
                                   alpha=alpha,
                                   UMAU=UMAU)
            intervals.append((self.variables[i], eta, 
                              (eta*self.Y).sum(), 
                              _interval))
        return intervals

    def model_pivots(self, which_step, alternative='greater'):
        """
        Compute two-sided pvalues for each coefficient
        in a given step of forward stepwise.

        Parameters
        ----------

        which_step : int
            Which step of forward stepwise.

        sigma : float
            Standard deviation of noise.

        Returns
        -------

        pivots : list
             List of (variable, pvalue)
             for selected model.

        """
        pivots = []
        LSfunc = np.linalg.pinv(self.X[:,self.variables[:which_step]])
        for i in range(LSfunc.shape[0]):
            pivots.append((self.variables[i],
                           self.constraints.pivot(LSfunc[i], self.Y,
                                                  alternative=alternative)))
        return pivots

    def model_quadratic(self, which_step):
        LSfunc = np.linalg.pinv(self.X[:,self.variables[:which_step]])
        P_LS = np.linalg.svd(LSfunc, full_matrices=False)[2]
        return quadratic_test(self.Y, P_LS, self.constraints)

def canonicalA(RX, RY, idx, sign, scale=None):
    """
    The canonical set of inequalities for a step of forward stepwise.
    These encode that 
    `sign*np.dot(RX.T[idx],RY)=np.fabs(np.dot(RX,RY)).max()` which is
    $\|RX^TRY\|_{\infty}$.

    Parameters
    ==========

    RX : `np.array((n,p))`

    RY : `np.array(n)`

    idx : `int`
        Maximizing index of normalized `np.fabs(np.dot(RX.T,RY))` where normalization
        is left multiplication by a diagonal matrix
        represented  by `scale` and is generally such that each row of `RX.T` has $\ell_2$
        norm of 1. 

    sign : `[-1,1]`

    scale : `np.array(p)`
        A diagonal matrix to apply before computing the $\ell_{\infty}$ norm.

    """

    n, p = RX.shape

    if scale is None:
        scale = np.ones(p)

    A0 = np.vstack([np.diag(1./scale), np.diag(-1./scale)])
    v = np.zeros(p)
    v[idx] = sign/scale[idx]
    A = v[None,:] - A0

    U = np.dot(A0, np.dot(RX.T, RY))
    if DEBUG:
        if sign > 0:
            print np.fabs(U).max(), U[idx], 'should match'
        else:
            print np.fabs(U).max(), U[idx+p], 'should match'

    keep = np.ones(2*p, np.bool)
    if sign > 0:
        keep[idx] = 0
    else:
        keep[idx+p] = 0

    A = A[keep]

    V = np.dot(A, RX.T)
    return -V

