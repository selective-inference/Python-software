"""
This module contains a class `lasso`_ that implements
post selection for the lasso
as described in `post selection LASSO`_.

It also includes a function `covtest`_ that computes
the exact form of the covariance test described in 
`Spacings`_.

The covariance test itself is asymptotically exponential
(under certain conditions) and is  described in 
`covTest`_.

.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238


"""
import numpy as np
from sklearn.linear_model import Lasso
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
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

       \text{minimize}_{\beta} \frac{1}{2n} \|y-X\beta\|^2_2 + 
            f \lambda_{\max} \|\beta\|_1

    where $f$ is `frac` and 

    .. math::

       \lambda_{\max} = \frac{1}{n} \|X^ty\|_{\infty}

    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    def __init__(self, y, X, frac=0.9, sigma_epsilon=1):


        self.y = y
        self.X = X
        self.frac = frac
        self.sigma_epsilon = sigma_epsilon
        n, p = X.shape
        self.lagrange = frac * np.fabs(np.dot(X.T, y)).max() / n
        self._covariance = self.sigma_epsilon**2 * np.identity(X.shape[0])

    def fit(self, sklearn_alpha=None, **lasso_args):
        """
        self.soln only updated after self.fit

        Parameters
        ----------

        sklearn_alpha : float
            Lagrange parameter, in the normalization set by `sklearn`.

        lasso_args : keyword args
             Passed to `sklearn.linear_model.Lasso`_

        Returns
        -------

        soln : np.float
             Solution to lasso with `sklearn_alpha=self.lagrange`.

        Notes
        -----

        Also calls `form_constraints`.

        """
        if sklearn_alpha is not None:
            self.lagrange = sklearn_alpha
        self._lasso = Lasso(alpha=self.lagrange, **lasso_args)
        self._lasso.fit(self.X, self.y)
        self._soln = self._lasso.coef_
        return self._soln

    def form_constraints(self):
        """
        After having fit lasso, form the constraints
        necessary for inference.

        This sets the attributes: `active_constraints`,
        `inactive_constraints`, `active`.
        """

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
                (-sA[:,None] * XAinv, 
                 -n*lagrange*sA*np.dot(self._SigmaA, 
                                         sA)), None)
            self._SigmaA *=  self.sigma_epsilon**2
            self._PA = PA = np.dot(XA, XAinv)
            irrep_subgrad = (n * lagrange * 
                             np.dot(np.dot(XnotA.T, XAinv.T), sA))

        else:
            XnotA = X
            self._PA = PA = 0
            self._XAinv = None
            irrep_subgrad = np.zeros(p)
            self.active_constraints = None

        if A.sum() < X.shape[1]:
            inactiveX = np.dot(np.identity(n) - PA, XnotA)
            scaling = np.ones(inactiveX.shape[1]) # np.sqrt((inactiveX**2).sum(0))
            inactiveX /= scaling[None,:]

            self.inactive_constraints = stack( 
                constraints((-inactiveX.T, 
                              lagrange * n + 
                              irrep_subgrad), None),
                constraints((inactiveX.T, 
                             lagrange * n -
                             irrep_subgrad), None))
        else:
            self.inactive_constraints = None

        if (self.active_constraints is not None 
            and self.inactive_constraints is not None):
            self._constraints = stack(self.active_constraints,
                                      self.inactive_constraints)
        elif self.active_constraints is not None:
            self._constraints = self.active_constraints
        else:
            self._constraints = self.inactive_constraints

    @property
    def soln(self):
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def constraints(self, doc="Constraint matrix for this LASSO problem"):
        if not hasattr(self, "_constraints"):
            self.form_constraints()
        return self._constraints

    @property
    def intervals(self, doc="OLS intervals for active variables adjusted for selection."):
        if not hasattr(self, "_intervals"):
            self._intervals = []
            C = self.constraints
            XAinv = self._XAinv
            if XAinv is not None:
                for i in range(XAinv.shape[0]):
                    eta = XAinv[i]
                    _interval = C.interval(eta, self.y,
                                           alpha=self.alpha,
                                           UMAU=self.UMAU)
                    self._intervals.append((self.active[i], eta, 
                                            (eta*self.y).sum(), 
                                            _interval))
        return self._intervals

    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
        " for selection."):
        if not hasattr(self, "_pvals"):
            self._pvals = []
            C = self.constraints
            XAinv = self._XAinv
            if XAinv is not None:
                for i in range(XAinv.shape[0]):
                    eta = XAinv[i]
                    _pval = C.pivot(eta, self.y)
                    _pval = 2 * min(_pval, 1 - _pval)
                    self._pvals.append((self.active[i], _pval))
        return self._pvals

    @property
    def unadjusted_intervals(self, doc="Unadjusted OLS intervals for active variables."):
        if not hasattr(self, "_intervals_unadjusted"):
            if not hasattr(self, "_constraints"):
                self.form_constraints()
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

    idx : int
        Variable achieving $\lambda_1$

    sign : int
        Sign of $X^Ty$ for variable achieving $\lambda_1$.

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

    return con, con.pivot(X[:,idx] * sign, Y, 'greater'), idx, sign

