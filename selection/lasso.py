import numpy as np
import regreg.api as rr
from .affine import (constraints, selection_interval,
                     interval_constraints,
                     stack)

from .variance_estimation import (interpolation_estimate,
                                  truncated_estimate)


from scipy.stats import norm as ndist
import warnings

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('cvx not available')
    pass

DEBUG = False

class lasso(object):

    """
    A class for the LASSO that performs the two tests,
    and returns selection intervals for the 
    active coefficients.

    WARNING: Changing `frac` will not propagate:
    constraints will not be updated, not soln, etc.
    
    Better to create a new instance.

    """
    
    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    def __init__(self, y, X, frac=0.9, sigma_epsilon=1):
        self.y = y
        self.X = X
        self.frac = frac
        self.sigma_epsilon = sigma_epsilon
        self.lagrange = frac * np.fabs(np.dot(X.T, y)).max()
        self._covariance = self.sigma_epsilon**2 * np.identity(X.shape[0])

    def fit(self, tol=1.e-8,
            min_its=50, use_cvx=False):
        """
        self.soln only updated after self.fit
        """

        if use_cvx: 
            beta = cvx.variable(p)
            _X = cvx.parameter(n,p)
            _X.value = cvx.matrix(X)

            _Y = cvx.parameter(n,1)
            _Y.value = cvx.matrix(y.reshape((n,1)))

            beta = cvx.variable(p)

            objective = cvx.sum(cvx.square(_Y-_X*beta))
            penalty = cvx.sum(cvx.abs(beta))
            program = cvx.program(cvx.minimize(0.5*objective + lagrange*penalty))
            program.solve(quiet=True)

            soln = np.array(beta.value).reshape(-1)
            soln[np.fabs(soln) < tol * np.fabs(soln).max()] = 0
        else: # use regreg
            X, y = self.X, self.y
            n, p = self.X.shape
            penalty = rr.l1norm(p, lagrange=self.lagrange)
            loss = rr.squared_error(X, y)
            problem = rr.simple_problem(loss, penalty)
            soln = problem.solve(tol=tol, min_its=min_its)
        self._soln = soln
        # evaluate properties -- bad form
        self.constraints
        return self._soln

    @property
    def soln(self):
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def centered_test(self):
        return fixed_pvalue_centered(self.y, 
                                     self.X, 
                                     self.lagrange, 
                                     self.soln, 
                                     sigma_epsilon=self.sigma_epsilon)[:2]

    @property
    def basic_test(self):
        return fixed_pvalue_uncentered(self.y, 
                                       self.X, 
                                       self.lagrange, 
                                       self.soln, 
                                       sigma_epsilon=self.sigma_epsilon)[:2]


    @property
    def constraints(self, doc="Constraint matrix for this LASSO problem"):
        if not hasattr(self, "_constraints"):
            X, y, soln, lagrange = self.X, self.y, self.soln, self.lagrange
            n, p = X.shape

            nonzero_coef = soln != 0
            tight_subgrad = np.fabs(np.fabs(np.dot(X.T, y - np.dot(X, soln))) / lagrange - 1) < 1.e-3
            if DEBUG:
                print 'KKT consistency', (nonzero_coef - tight_subgrad).sum()

            A = nonzero_coef
            self.active = np.nonzero(nonzero_coef)[0]
            if A.sum() > 0:
                sA = np.sign(soln[A])
                self.signs = sA
                XA = X[:,A]
                XnotA = X[:,~A]
                self._XAinv = XAinv = np.linalg.pinv(XA)
                self._SigmaA = np.dot(XAinv, XAinv.T)

                self.active_constraints = constraints(  
                    (sA[:,None] * XAinv, 
                     self.lagrange*sA*np.dot(self._SigmaA, 
                                             sA)), None)
                self._SigmaA *=  self.sigma_epsilon**2
                self.PA = PA = np.dot(XA, XAinv)
                irrep_subgrad = lagrange * np.dot(np.dot(XnotA.T, XAinv.T), sA)

            else:
                XnotA = X
                self.PA = PA = 0
                irrep_supgrad = np.zeros(p)
                self.active_constraints = None

            if A.sum() < X.shape[1]:
                inactiveX = np.dot(np.identity(n) - PA, XnotA)
                scaling = np.sqrt((inactiveX**2).sum(0))
                inactiveX /= scaling[None,:]

                self.inactive_constraints = stack( 
                    constraints(((-inactiveX.T, 
                                   -(lagrange - 
                                    irrep_subgrad) / scaling)), None),
                    constraints(((inactiveX.T, 
                                   -(lagrange +
                                    irrep_subgrad) / scaling)), None))
            else:
                self.inactive_constraints = None

            #_constraints = active_constraints + inactive_constraints
            #_linear_part = np.vstack([A for A, _ in _constraints])
            #_offset_part = np.hstack([b for _, b in _constraints])
            if (self.active_constraints is not None 
                and self.inactive_constraints is not None):
                self._constraints = stack(self.active_constraints,
                                          self.inactive_constraints)
            elif self.active_constraints is not None:
                self._constraints = self.active_constraints
            else:
                self._constraints = self.inactive_constraints


        return self._constraints

    @property
    def intervals(self, doc="OLS intervals for active variables adjusted for selection."):
        if not hasattr(self, "_intervals"):
            self._intervals = []
            C = self.constraints
            XAinv = self._XAinv
            for i in range(XAinv.shape[0]):
                eta = XAinv[i]
                _interval = selection_interval( \
                       C.inequality,
                       C.inequality_offset,
                       self._covariance,
                       self.y,
                       eta,
                       alpha=self.alpha,
                       UMAU=self.UMAU)
                self._intervals.append((self.active[i], eta, (eta*self.y).sum(), 
                                        _interval))
        return self._intervals

    @property
    def l1norm_interval(self, doc="Interval for $s^T\beta_E(\mu)$."):
        if not hasattr(self, "_l1interval"):
            self._intervals = []
            C = self.constraints
            XAinv = self._XAinv
            eta = np.dot(self.signs, XAinv)
            if DEBUG:
                print interval_constraints( \
                    C.inequality,
                    C.inequality_offset,
                    self._covariance,
                    self.y,
                    eta)[:3]

            self._l1interval = selection_interval( \
                C.inequality,
                C.inequality_offset,
                self._covariance,
                self.y,
                eta,
                alpha=self.alpha)
        return self._l1interval

    @property
    def active_pvalues(self, doc="OLS intervals for active variables adjusted for selection."):
        if not hasattr(self, "_pvals"):
            self._pvals = []
            C = self.constraints
            XAinv = self._XAinv
            for i in range(XAinv.shape[0]):
                eta = XAinv[i]
                _pval = C.pivot(eta, self.y)
                _pval = 2 * min(_pval, 1 - _pval)
                self._pvals.append((self.active[i], _pval))
        return self._pvals

    @property
    def unadjusted_intervals(self, doc="Unadjusted OLS intervals for active variables."):
        if not hasattr(self, "_intervals_unadjusted"):
            self.constraints # force self._SigmaA to be computed -- 
                             # bad use of property
            self._intervals_unadjusted = []
            XAinv = self._XAinv
            for i in range(self.active.shape[0]):
                eta = XAinv[i]
                center = (eta*self.y).sum()
                width = ndist.ppf(1-self.alpha/2.) * np.sqrt(self._SigmaA[i,i])
                _interval = [center-width, center+width]
                self._intervals_unadjusted.append((self.active[i], eta, (eta*self.y).sum(), 
                                        _interval))
        return self._intervals_unadjusted

class FixedLambdaError(ValueError):
    pass

def estimate_sigma(y, X, frac=0.1, 
                   lower=0.5,
                   upper=2,
                   npts=15,
                   ndraw=5000,
                   burnin=1000):
    """
    Estimate the parameter $\sigma$ in $y \sim N(X\beta, \sigma^2 I)$
    after fitting LASSO with Lagrange parameter `frac` times
    $\lambda_{\max}=\|X^Ty\|_{\infty}$.

    Uses `selection.variance_estimation.interpolation_estimate`

    Parameters
    ----------

    y : np.float
        Response to be used for LASSO.

    X : np.float
        Design matrix to be used for LASSO.

    frac : float
        What fraction of $\lambda_{\max}$ should be used to fit
        LASSO.

    lower : float
        Multiple of naive estimate to use as lower endpoint.

    upper : float
        Multiple of naive estimate to use as upper endpoint.

    npts : int
        Number of points in interpolation grid.

    ndraw : int
        Number of Gibbs steps to use for estimating
        each expectation.

    burnin : int
        How many Gibbs steps to use for burning in.

    Returns
    -------

    sigma_hat : float
        The root of the interpolant derived from GCM values.

    interpolaint : scipy.interpolate.interp1d
        The interpolant, to be used for plotting or other 
        diagnostics.

    """

    n, p = X.shape
    L = lasso(y, X, frac=frac)
    soln = L.fit(tol=1.e-14, min_its=200)

    # now form the constraint for the inactive variables

    C = L.inactive_constraints
    PR = np.identity(n) - L.PA
    try:
        U, D, V = np.linalg.svd(PR)
    except np.linalg.LinAlgError:
        D, U = np.linalg.eigh(PR)

    keep = D >= 0.5
    U = U[:,keep]
    Z = np.dot(U.T, y)
    Z_inequality = np.dot(C.inequality, U)
    Z_constraint = constraints((Z_inequality, C.inequality_offset), None)
    if not Z_constraint(Z):
        raise ValueError('Constraint not satisfied. Gibbs algorithm will fail.')
    return interpolation_estimate(Z, Z_constraint,
                                  lower=lower,
                                  upper=upper,
                                  npts=npts,
                                  ndraw=ndraw,
                                  burnin=burnin,
                                  estimator='simulate')

def covtest(X, Y, sigma=1):
    """
    The exact form of the covariance test, described
    in the `Kac Rice`_ and `Spacings`_ papers.

    .. _Kac Rice: http://arxiv.org/abs/1308.3020
    .. _Spacings: http://arxiv.org/abs/1401.3889

    Parameters
    ----------

    X : np.float((n,p))

    Y : np.float(n)

    sigma : float

    Returns
    -------

    con : `selection.constraints.constraints`_
        The constraint based on conditioning
        on the sign and location of the maximizer.

    pvalue : float
        Exact covariance test p-value.

    """
    n, p = X.shape

    Z = np.dot(X.T, Y)
    idx = np.argsort(np.fabs(Z))[-1]
    sign = np.sign(Z[idx])

    I = np.identity(p)
    subset = np.ones(p, np.bool)
    subset[idx] = 0
    selector = np.vstack([X.T[subset],-X.T[subset]])
    selector -= (sign * X[:,idx])[None,:]

    con = constraints((selector, np.zeros(selector.shape[0])),
                      None)

    return con, con.pivot(X[:,idx] * sign, Y, 'greater')

