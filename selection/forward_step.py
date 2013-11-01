"""
In this module, we implement forward stepwise model selection for $K$ steps.

The main goal of this is to produce a set of linear inequality constraints satisfied by
$y$ after $K$ steps.

"""

import numpy as np
import scipy.sparse
from scipy.stats import norm as ndist

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

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.P = [None] # residual forming projections
        self.A = None
        self.variables = []
        self.signs = []
        self.covariance = np.identity(X.shape[0])

    def step(self):
        """
        Take one step of forward stepwise.
        Internally, this has the effect of: 

        * adding a new (lowrank) projection to `self.P`, 
        
        * adding a new variable to `self.variables`

        * adding a certain number of rows to `self.A`

        * signs are also tracked (unnecessarily for the moment) in `self.signs`

        The multiplication `np.dot(self.A, gamma)` can be made more 
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
            self.A = canonicalA(X, Y, idx, sign, scale=scale)
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
        return np.dot(self.A, self.Y).min() > 0

    def bounds(self, gamma):
        """
        Find implied upper and lower limits for a given
        direction of interest.

        Parameters
        ==========

        gamma : `np.array(n)`

        Returns
        =======

        Mplus: float
             Lower bound for $\gamma^TY$ for cone determined by `self`.

        V : float
             The center $\gamma^TY$.

        Mminus : float
             Lower bound for $\gamma^TY$ for cone determined by `self`.

        sigma : float
             $\ell_2$ norm of `gamma` (assuming `self.covariance` is $I$)
        """
        return interval_constraints(self.A,
                                    np.zeros(self.A.shape[0]),
                                    self.covariance,
                                    self.Y,
                                    gamma)

def canonicalA(RX, RY, idx, sign, scale=None):
    """
    The canonical set of inequalities for a step of forward stepwise.
    These encode that `sign*np.dot(RX.T[idx],RY)=np.fabs(np.dot(RX,RY)).max()` which is
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
    return V

def interval_constraints(support_directions, 
                         support_offsets,
                         covariance,
                         observed_data, 
                         direction_of_interest,
                         tol = 1.e-4):
    """
    Given an affine cone constraint $Ax+b \geq 0$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $w$, and
    an observed Gaussian vector $Z$ with some `covariance`, this
    function returns $w^TZ$ as well as an interval
    bounding this value. 

    The interval constructed is such that the endpoints are 
    independent of $w^TZ$, hence the $p$-value
    of `Kac-Rice <http://arxiv.org/abs/1308.3020>`_
    can be used to form an exact pivot.

    Parameters
    ==========

    support_directions : `np.array((q,n))`

    support_offsets : `np.array(q)`

    covariance : `np.array((n,n))`
        Covariance of $Z$.

    observed_data : `np.array(n)`
        The observed $Z$.

    direction_of_interest : `np.array(n)`
        For what combination of $Z$ do we want bounds?

    Returns
    =======

        Mplus: float
             Lower bound for $\gamma^TY$ for cone determined by `self`.

        V : float
             The center $\gamma^TY$.

        Mminus : float
             Lower bound for $\gamma^TY$ for cone determined by `self`.

        sigma : float
             The covariance of this linear combination of $Z$: $\sqrt{\gamma^T S\gamma}$.

    """

    # shorthand
    A, b, S, X, w = (support_directions,
                     support_offsets,
                     covariance,
                     observed_data,
                     direction_of_interest)

    U = np.dot(A, X) + b
    if not np.all(U > -tol * np.fabs(U).max()):
        warn('constraints not satisfied: %s' % `U`)

    Sw = np.dot(S, w)
    sigma = np.sqrt((w*Sw).sum())
    C = np.dot(A, Sw) / sigma**2
    V = (w*X).sum()
    RHS = (-b - np.dot(A, X) + V * C) / C
    pos_coords = C > tol * np.fabs(C).max()
    if np.any(pos_coords):
        lower_bound = RHS[pos_coords].max()
    else:
        lower_bound = -np.inf
    neg_coords = C < -tol * np.fabs(C).max()
    if np.any(neg_coords):
        upper_bound = RHS[neg_coords].min()
    else:
        upper_bound = np.inf

    return lower_bound, V, upper_bound, sigma


if __name__ == "__main__":

    n, p = 100, 40
    X = np.random.standard_normal((n,p))
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100)
    
    FS = forward_stepwise(X, Y)
    
    for i in range(30):
        FS.step()
        if not FS.check_constraints():
            raise ValueError('constraints not satisfied')

    print 'first 30 variables selected', FS.variables

    print 'M^{\pm} for the 10th selected model knowing that we performed 30 steps of forward stepwise'

    LSfunc = np.linalg.pinv(X[:,FS.variables[:10]])
    for i in range(LSfunc.shape[0]):
        print FS.bounds(LSfunc[i])
