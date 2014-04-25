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

    This can be enforced by calling the `orthogonalize` method.
    
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

    def __init__(self, X, Y, sigma=1.,
                 subset=None):
        self.sigma = sigma
        if subset is None:
            subset = np.ones(Y.shape[0], np.bool)
        self.X = X[subset]
        self.Y = Y[subset]
        self.P = [None] # residual forming projections
        self.A = None
        self.variables = []
        self.signs = []
        self.covariance = self.sigma**2 * np.identity(self.X.shape[0])

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
            idx = np.argmax(np.fabs(U) / scale)
            sign = np.sign(U[idx])
            Unew = X[:,idx] / scale[idx]
            Pnew = projection(Unew.reshape((-1,1)))
            self.As = [canonicalA(X, Y, idx, sign, scale=scale)]
            self.A = self.As[0]
            self.variables.append(idx)
            self.signs.append(sign)
        else:
            RY = Y-P(Y)
            RX = X-P(X)
            keep = np.ones(p, np.bool)
            keep[self.variables] = 0
            RX = RX[:,keep]

            scale = np.sqrt((RX**2).sum(0))
            U = np.dot(RX.T, RY)
            idx = np.argmax(np.fabs(U) / scale)

            sign = np.sign(U[idx])
            self.variables.append(np.arange(p)[keep][idx])
            self.signs.append(sign)
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
            print np.dot(self.A, Y).min(), 'should be nonnegative'

        self.P.append(Pnew)

    def check_constraints(self):
        """
        Verify whether or not constraints are consistent with `self.Y`.
        """
        return np.dot(self.A, self.Y).max() <= 0

    def bounds(self, eta):
        """
        Find implied upper and lower limits for a given
        direction of interest.

        Parameters
        ==========

        eta : `np.array(n)`

        Returns
        =======

        Mplus: float
             Lower bound for $\eta^TY$ for cone determined by `self`.

        V : float
             The center $\eta^TY$.

        Mminus : float
             Lower bound for $\eta^TY$ for cone determined by `self`.

        sigma : float
             $\ell_2$ norm of `eta` (assuming `self.covariance` is $I$)
        """

        return self.constraints.pivots(eta, self.Y)

    @property
    def constraints(self):
        return constraints((self.A, np.zeros(self.A.shape[0])), None,
                           covariance=self.covariance)

    # pivots we might care about

    def model_pivots(self, which_step):
        pivots = []
        LSfunc = np.linalg.pinv(self.X[:,self.variables[:which_step]])
        for i in range(LSfunc.shape[0]):
            pivots.append(self.bounds(LSfunc[i]))
        return pivots

    def model_quadratic(self, which_step):
        LSfunc = np.linalg.pinv(self.X[:,self.variables[:which_step]])
        P_LS = np.linalg.svd(LSfunc, full_matrices=False)[2]
        return quadratic_test(self.Y / self.sigma, P_LS, self.constraints)

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

