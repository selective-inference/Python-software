"""
This module contains a class `lasso`_ that implements
post selection for the lasso
as described in `post selection LASSO`_.


.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238


"""
import numpy as np
from sklearn.linear_model import Lasso
from .affine import (constraints, selection_interval,
                     interval_constraints,
                     sample_from_constraints,
                     stack)
from .discrete_family import discrete_family

from scipy.stats import norm as ndist
import warnings

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('cvx not available')
    pass

DEBUG = False


class lasso(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \frac{1}{2n} \|y-X\beta\|^2_2 + 
            f \lambda_{\max} \|\beta\|_1

    where $f$ is `frac` and 

    .. math::

       \lambda_{\max} = \frac{1}{n} \|X^Ty\|_{\infty}

    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    def __init__(self, y, X, lam, sigma=1):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        y : np.float(y)
            The target, in the model $y = X\beta$

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        lam : np.float
            Coefficient of the L-1 penalty in
            $\text{minimize}_{\beta} \frac{1}{2} \|y-X\beta\|^2_2 + 
                \lambda\|\beta\|_1$

        sigma : np.float
            Standard deviation of the gaussian distribution :
            The covariance matrix is
            `sigma**2 * np.identity(X.shape[0])`.
            Defauts to 1.
        """
        self.y = y
        self.X = X
        self.sigma = sigma
        n, p = X.shape
        self.lagrange = lam / n
        self._covariance = self.sigma**2 * np.identity(X.shape[0])

    def fit(self, sklearn_alpha=None, **lasso_args):
        """
        Fit the lasso using `Lasso` from `sklearn`.
        This sets the attribute `soln` and
        forms the constraints necessary for post-selection inference
        by caling `form_constraints()`.

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
             
        
        """

        # fit Lasso using scikit-learn
        
        clf = Lasso(alpha = self.lagrange, fit_intercept = False)
        clf.fit(self.X, self.y)
        self._soln = beta = clf.coef_       
        self.form_constraints()
        
    def form_constraints(self):
        """
        After having fit lasso, form the constraints
        necessary for inference.

        This sets the attributes: `active_constraints`,
        `inactive_constraints`, `active`.

        Returns
        -------

        None

        """

        # determine equicorrelation set and signs
        beta = self.soln
        n, p = self.X.shape
        lam = self.lagrange * n

        active = (beta != 0)
        self.z_E = np.sign(beta[active])

        # calculate the "partial correlation" operator R = X_{-E}^T (I - P_E)
        X_E = self.X[:,active]
        X_notE = self.X[:,~active]
        self._XEinv = np.linalg.pinv(X_E)
        P_E = np.dot(X_E, self._XEinv)
        R = np.dot(X_notE.T, np.eye(n)-P_E)
        self.active = np.nonzero(active)[0]

        (self._active_constraints, 
         self._inactive_constraints, 
         self._constraints) = _constraint_from_data(X_E, X_notE, self.z_E, active, lam, self.sigma, R)

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def active_constraints(self):
        """
        Affine constraints imposed on the
        active variables by the KKT conditions.
        """
        return self._active_constraints

    @property
    def inactive_constraints(self):
        """
        Affine constraints imposed on the
        inactive subgradient by the KKT conditions.
        """
        return self._inactive_constraints

    @property
    def constraints(self):
        """
        Affine constraints for this LASSO problem.
        This is `self.active_constraints` stacked with
        `self.inactive_constraints`.
        """
        return self._constraints

    @property
    def intervals(self):
        """
        Intervals for OLS parameters of active variables
        adjusted for selection.

        
        """
        if not hasattr(self, "_intervals"):
            self._intervals = []
            C = self.active_constraints
            XEinv = self._XEinv
            if XEinv is not None:
                for i in range(XEinv.shape[0]):
                    eta = XEinv[i]
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
            C = self.active_constraints
            XEinv = self._XEinv
            if XEinv is not None:
                for i in range(XEinv.shape[0]):
                    eta = XEinv[i]
                    _pval = C.pivot(eta, self.y)
                    _pval = 2 * min(_pval, 1 - _pval)
                    self._pvals.append((self.active[i], _pval))
        return self._pvals

    @property
    def nominal_intervals(self):
        """
        Intervals for OLS parameters of active variables
        that have not been adjusted for selection.
        """
        if not hasattr(self, "_intervals_unadjusted"):
            if not hasattr(self, "_constraints"):
                self.form_constraints()
            self._intervals_unadjusted = []
            XEinv = self._XEinv
            SigmaE = self.sigma**2 * np.dot(XEinv, XEinv.T)
            for i in range(self.active.shape[0]):
                eta = XEinv[i]
                center = (eta*self.y).sum()
                width = ndist.ppf(1-self.alpha/2.) * np.sqrt(SigmaE[i,i])
                _interval = [center-width, center+width]
                self._intervals_unadjusted.append((self.active[i], eta, (eta*self.y).sum(), 
                                        _interval))
        return self._intervals_unadjusted
def _constraint_from_data(X_E, X_notE, z_E, E, lam, sigma, R):

    n, p = X_E.shape[0], X_E.shape[1] + X_notE.shape[1]
    if np.array(lam).shape == ():
        lam = np.ones(p) * lam

    # inactive constraints
    A0 = np.vstack((R, -R)) / np.hstack([lam[~E],lam[~E]])[:,None]
    b_tmp = np.dot(X_notE.T, np.dot(np.linalg.pinv(X_E.T), z_E))
    b0 = np.concatenate((1.-b_tmp, 1.+b_tmp))
    _inactive_constraints = constraints(A0, b0)
    _inactive_constraints.covariance *= sigma**2

    # active constraints
    C = np.linalg.inv(np.dot(X_E.T, X_E))
    A1 = -np.dot(np.diag(z_E), np.dot(C, X_E.T))
    b1 = -np.dot(np.diag(z_E), np.dot(C, z_E))*lam[E]

    _active_constraints = constraints(A1, b1)
    _active_constraints.covariance *= sigma**2

    _constraints = stack(_active_constraints,
                         _inactive_constraints)
    _constraints.covariance *= sigma**2
    return _active_constraints, _inactive_constraints, _constraints

def standard_lasso(y, X, sigma=1, lam_frac=1.):
    """
    Fit a LASSO with a default choice of Lagrange parameter
    equal to `lam_frac` times $\sigma \cdot E(|X^T\epsilon|)$
    with $\epsilon$ IID N(0,1).

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    lam_frac : float
        Multiplier for choice of $\lambda$

    Returns
    -------

    lasso_selection : `lasso`
         Instance of `lasso` after fitting. 

    """
    n, p = X.shape

    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 50000)))).max(0))

    lasso_selector = lasso(y, X, lam, sigma=sigma)
    lasso_selector.fit()
    return lasso_selector

def data_carving(y, X, sigma=1, lam_frac=1.,
                 split_frac=0.9,
                 beta_parameter_sampling=None,
                 ndraw=80000, burnin=20000):

    """
    Fit a LASSO with a default choice of Lagrange parameter
    equal to `lam_frac` times $\sigma \cdot E(|X^T\epsilon|)$
    with $\epsilon$ IID N(0,1) on a proportion (`split_frac`) of
    the data.

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    lam_frac : float
        Multiplier for choice of $\lambda$

    split_frac : float
        What proportion of the data to use in the first stage?

    burnin : int
        How many burnin samples for Gibbs hit-and-run sampler.

    ndraw : int
        How many draws to keep from Gibbs hit-and-run sampler.
    Returns
    -------

    full_con : `constraints`
         Constraints on all data after 

    pvalues : [(variable_id, float)]
         One sided tests for each selected variable
         with signs chosen from the LASSO selected sign.
         
    intervals : [(variable_id, float, float)]
         Selection intervals for each selected variable.
         Intervals are the equal-tailed intervals.

    """

    n, p = X.shape
    splitn = int(n*split_frac)
    indices = np.arange(n)
    np.random.shuffle(indices)
    stage_one = indices[:splitn]
    y1, X1 = y[stage_one], X[stage_one]
    
    first_stage_selector = standard_lasso(y1, X1, sigma=sigma, lam_frac=lam_frac)
    selector = np.identity(n)[stage_one]
    linear_part = np.dot(first_stage_selector.constraints.linear_part,
                         selector)
    full_con = constraints(linear_part, 
                           first_stage_selector.constraints.offset,
                           covariance=sigma**2 * np.identity(n))

    active = first_stage_selector.active
    Xa_inv = np.linalg.pinv(X[:,active])

    full_con.mean = np.dot(X[:, active], np.dot(Xa_inv, y))
    beta_OLS = np.dot(Xa_inv, full_con.mean)
    sign_beta = np.sign(np.dot(np.linalg.pinv(X1[:, active]), y1))

    pvalues = []
    intervals = []

    for i, a in enumerate(active):
        keep = np.zeros(p, np.bool)
        keep[active] = 1
        keep[a] = 0

        eta = Xa_inv[i] * sign_beta[i]
        natural_parameter_sampling = beta_OLS[i] * sign_beta[i] / (sigma**2 * (eta**2).sum())
        observed = (y*eta).sum()
        conditional_con = full_con.conditional(X[:,keep].T, 
                                               np.dot(X[:,keep].T, y))
        Z = sample_from_constraints(full_con,
                                    y,
                                    eta,
                                    ndraw=ndraw,
                                    burnin=burnin)
        null_statistics = np.dot(Z, eta)

        # these weights
        # retilt the distribution
        # back to the null

        logW = -natural_parameter_sampling * null_statistics 
        logW -= logW.max() - 2.
        W = np.exp(logW)

        # pvalues and intervals
        # can be found from a discrete 
        # exponential family

        family = discrete_family(null_statistics, W)
        pval = family.ccdf(0, observed)
        pvalues.append([a,pval])
        
        interval = family.equal_tailed_interval(observed)
        # this is an interval for the natural parameter which is
        # \eta^T\mu / \|\eta\|^2 \sigma^2

        interval = (interval[0] * (eta**2).sum() * sigma**2,
                    interval[1] * (eta**2).sum() * sigma**2)
        intervals.append([a, interval[0], interval[1]])

    return full_con, pvalues, intervals, first_stage_selector, sign_beta
