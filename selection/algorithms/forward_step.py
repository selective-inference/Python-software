"""
In this module, we implement forward stepwise model selection for $K$ steps.

The main goal of this is to produce a set of linear inequality constraints satisfied by
$y$ after $K$ steps.

"""

from copy import copy

import numpy as np

# local imports 

from ..constraints.affine import constraints, gibbs_test, stack
from ..distributions.chisq import quadratic_test
from .projection import projection

DEBUG = False

class forward_stepwise(object):

    """
    Centers columns of X!
    """

    def __init__(self, X, Y, 
                 subset=[],
                 covariance=None):
        self.subset = subset
        self.X, self.Y = X, Y
        if subset != []:
            self.Xsub = X.copy()[subset]
            self.Xsub -= self.Xsub.mean(0)[None,:]
            self.Ysub = Y.copy()[subset]
            self.Ysub -= self.Ysub.mean()
            self.subset_selector = np.identity(self.X.shape[0])[subset]
        else:
            self.Xsub = X.copy()
            self.Ysub = Y.copy()
        self.P = [None] # residual forming projections
        self.A = None
        self.variables = []
        self.Z = []
        self.Zfunc = []
        self.signs = []
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
        
        X, Y = self.Xsub, self.Ysub
        n, p = self.Xsub.shape

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
            self.Zfunc.append(Unew * sign)
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
            self.Zfunc.append((RX.T[idx] / scale[idx]) * sign)
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
        if not hasattr(self, "_constraints"):
            if self.subset == []:
                self._constraints = constraints(self.A, np.zeros(self.A.shape[0]), 
                                                covariance=self.covariance)
            else:
                self._constraints = constraints(np.dot(self.A, self.subset_selector),
                                                np.zeros(self.A.shape[0]), 
                                                covariance=self.covariance)
        return self._constraints

    def model_intervals(self, which_step, alpha=0.05, UMAU=False):
        """
        Compute selection intervals for
        a given step of forward stepwise 
        using saturated model.

        Parameters
        ----------

        which_step : int
            Which step of forward stepwise.

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

    def model_pivots(self, which_step, alternative='greater',
                     saturated=True,
                     ndraw=5000,
                     burnin=2000,
                     which_var=[]):
        """
        Compute two-sided pvalues for each coefficient
        in a given step of forward stepwise.

        Parameters
        ----------

        which_step : int
            Which step of forward stepwise.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        saturated : bool
            Use saturated model or selected model?

        ndraw : int (optional)
            Defaults to 5000.

        burnin : int (optional)
            Defaults to 2000.

        which_var : []
            Compute pivots for which variables? If empty,
            return a pivot for all selected variable at stage `which_step`.

        Returns
        -------

        pivots : list
             List of (variable, pvalue)
             for selected model.

        """

        if which_step == 0:
            return []

        if self.covariance is None and saturated:
            raise ValueError('need a covariance matrix to compute pivots for saturated model')

        con = copy(self.constraints)
        if self.covariance is not None:
            con.covariance[:] = self.covariance 

        linear_part = self.X[:,self.variables[:which_step]]
        observed = np.dot(linear_part.T, self.Y)
        LSfunc = np.linalg.pinv(linear_part)

        if which_var == []:
            which_var = self.variables[:which_step]

        pivots = []
        if saturated:
            for i in range(LSfunc.shape[0]):
                if self.variables[i] in which_var:
                    pivots.append((self.variables[i],
                                   con.pivot(LSfunc[i], self.Y,
                                             alternative=alternative)))
        else:
            sigma_known = self.covariance is not None
            for i in range(LSfunc.shape[0]):
                if self.variables[i] in which_var:
                    keep = np.ones(LSfunc.shape[0], np.bool)
                    keep[i] = False

                    if which_step > 1:
                        conditional_con = con.conditional(linear_part.T[keep],
                                                          observed[keep])
                    else:
                        conditional_con = con
                    pval = gibbs_test(conditional_con,
                                      self.Y,
                                      LSfunc[i],
                                      alternative=alternative,
                                      sigma_known=sigma_known,
                                      burnin=burnin,
                                      ndraw=ndraw,
                                      how_often=50)[0]
                    pivots.append((self.variables[i], 
                                   pval))
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
    ----------

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

def info_crit_stop(X, Y, sigma, cost=2):
    """
    Fit model using forward stepwise,
    stopping using a rule like AIC or BIC.
    
    The error variance must be supplied, in which
    case AIC is essentially Mallow's C_p.

    Parameters
    ----------

    X : np.float
        Design matrix

    Y : np.float
        Response vector

    sigma : float (optional)
        Error variance.

    cost : float
        Cost per parameter. For BIC use cost=log(X.shape[0])

    Returns
    -------

    FS : `forward_stepwise`
        Instance of forward stepwise stopped at the
        corresponding step. Constraints of FS
        will reflect the minimum Z score requirement.

    """
    n, p = X.shape
    FS = forward_stepwise(X, Y, covariance=sigma**2 * np.identity(n))
    while True:
        FS.next()

        if FS.Z[-1] < sigma * np.sqrt(cost):
            break

    new_linear_part = -np.array(FS.Zfunc)
    new_linear_part[-1] *= -1
    new_offset = -sigma * np.sqrt(cost) * np.ones(new_linear_part.shape[0])
    new_offset[-1] *= -1

    new_con = stack(FS.constraints, constraints(new_linear_part, 
                                                new_offset))
    new_con.covariance[:] = sigma**2 * np.identity(n)
    if DEBUG:
        print FS.constraints.linear_part.shape, 'before'
    FS._constraints = new_con
    if DEBUG:
        print FS.constraints.linear_part.shape, 'should have added number of steps constraints'
    return FS

def sequential(X, Y, sigma=None, nstep=10,
               saturated=False,
               ndraw=5000,
               burnin=2000,
               subset=[]):
    """
    Fit model using forward stepwise,
    stopping using a rule like AIC or BIC.
    
    The error variance must be supplied, in which
    case AIC is essentially Mallow's C_p.

    Parameters
    ----------

    X : np.float
        Design matrix

    Y : np.float
        Response vector

    sigma : float (optional)
        Error variance.

    nstep : int
        How many steps should we take?

    saturated : bool
        Should we compute saturated or selected model pivots?

    ndraw : int (optional)
        Defaults to 5000.

    burnin : int (optional)
        Defaults to 2000.

    subset : []
        Subset of cases to use for selection, defaults to [].

    Returns
    -------

    FS : `forward_stepwise`
        Instance of forward stepwise after `nstep` steps.

    pvalues : []
        P-values computed at each step.

    """

    n, p = X.shape
    if sigma is not None:
        FS = forward_stepwise(X, Y, covariance=sigma**2 * np.identity(n),
                              subset=subset)
    else:
        FS = forward_stepwise(X, Y)

    pvalues = []
    for i in range(nstep):
        FS.next()
        pvalues.extend(FS.model_pivots(i+1, which_var=[FS.variables[-1]],
                                       saturated=saturated,
                                       ndraw=ndraw,
                                       burnin=burnin))
    return FS, pvalues

