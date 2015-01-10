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
                     gibbs_test,
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

def instance(n=100, p=200, s=7, sigma=5, rho=0.3, snr=7,
             random_signs=False):
    """
    A testing instance for the LASSO.
    Design is equi-correlated in the population,
    normalized to have columns of norm 1.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    sigma : float
        Noise level

    rho : float
        Equicorrelation value (must be in interval [0,1])

    snr : float
        Size of each coefficient

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    """

    X = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
        np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    X -= X.mean(0)[None,:]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    beta = np.zeros(p) 
    beta[:s] = snr 
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True
    Y = (np.dot(X, beta) + np.random.standard_normal(n)) * sigma
    return X, Y, beta, np.nonzero(active)[0], sigma

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

def data_carving(y, X, 
                 sigma=1, 
                 lam_frac=1.,
                 coverage=0.95, 
                 split_frac=0.9,
                 ndraw=5000,
                 burnin=1000,
                 splitting=False):

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
        Multiplier for choice of $\lambda$. Defaults to 2.

    coverage : float
        Coverage for selective intervals. Defaults to 0.95.

    split_frac : float (optional)
        What proportion of the data to use in the first stage?
        Defaults to 0.9.

    ndraw : int (optional)
        How many draws to keep from Gibbs hit-and-run sampler.
        Defaults to 5000.

    burnin : int (optional)
        Defaults to 1000.

    splitting : bool (optional)
        If True, also return splitting pvalues and intervals.

    Returns
    -------

    results : [(variable_id, pvalue, interval)]
        Identity of active variables with associated
        selected (twosided) pvalue and selective interval.

    """

    n, p = X.shape
    splitn = int(n*split_frac)
    indices = np.arange(n)
    np.random.shuffle(indices)
    stage_one = indices[:splitn]

    y1, X1 = y[stage_one], X[stage_one]

    first_stage_selector = L = standard_lasso(y1, X1, sigma=sigma, lam_frac=lam_frac)

    # quantities related to models fit on
    # stage_one and full dataset

    X_E = X[:,L.active]
    X_Ei = np.linalg.pinv(X_E)
    X_E1 = X1[:,L.active]
    X_Ei1 = np.linalg.pinv(X_E1)

    info_E = sigma**2 * np.dot(X_Ei, X_Ei.T)
    info_E1 = sigma**2 * np.dot(X_Ei1, X_Ei1.T)

    s = sparsity = L.active.shape[0]
    beta_E = np.dot(X_Ei, y)
    beta_E1 = np.dot(X_Ei1, y[stage_one])

    # setup the constraint on the 2s Gaussian vector

    linear_part = np.zeros((s, 2*s))
    linear_part[:, s:] = -np.diag(L.z_E)
    b = first_stage_selector.active_constraints.offset
    con = constraints(linear_part, b)

    # specify covariance of 2s Gaussian vector

    cov = np.zeros((2*s, 2*s))
    cov[:s, :s] = info_E
    cov[s:, :s] = info_E
    cov[:s, s:] = info_E
    cov[s:, s:] = info_E1

    con.covariance[:] = cov

    # how do we sample at the right beta?
    #con.mean = mu = np.hstack([beta_E, beta_E1])
    #weight_mu = np.dot(np.linalg.pinv(cov), mu)

    # for the conditional law
    # we will change the linear function for each coefficient

    selector = np.zeros((s, 2*s))
    selector[:, :s]  = np.identity(s)
    conditional_linear = np.dot(np.linalg.pinv(info_E), selector) * sigma**2

    # a valid initial condition

    initial = np.hstack([beta_E, beta_E1])

    pvalues = []
    intervals = []

    if splitting:
        stage_two = indices[splitn:]
        y2, X2 = y[stage_two], X[stage_two]
        X_E2 = X2[:,L.active]
        X_Ei2 = np.linalg.pinv(X_E2)
        beta_E2 = np.dot(X_Ei2, y2)
        info_E2 = np.dot(X_Ei2, X_Ei2.T) * sigma**2

        splitting_pvalues = []
        splitting_intervals = []

        split_cutoff = np.fabs(ndist.ppf((1. - coverage) / 2))

    # compute p-values and (TODO: intervals)

    for j in range(X_E.shape[1]):

        keep = np.ones(s, np.bool)
        keep[j] = 0

        eta = np.zeros(2*s)
        eta[j] = 1.

        conditional_law = con.conditional(conditional_linear[keep], \
                              np.dot(X_E.T, y)[keep])

        pval, Z, W = gibbs_test(conditional_law,
                                initial,
                                eta,
                                UMPU=False,
                                sigma_known=True,
                                ndraw=ndraw,
                                burnin=burnin,
                                how_often=5,
                                alternative='twosided')

        #W *= np.exp(-np.dot(Z, weight_mu))

        pvalues.append(pval)

        # intervals are still not implemented yet
        intervals.append(None)

        if splitting:
            split_pval = ndist.cdf(beta_E2[j] / np.sqrt(info_E2[j,j]))
            split_pval = 2 * min(split_pval, 1. - split_pval)
            splitting_pvalues.append(split_pval)

            splitting_interval = (beta_E2[j] - 
                                  split_cutoff * np.sqrt(info_E2[j,j]),
                                  beta_E2[j] + 
                                  split_cutoff * np.sqrt(info_E2[j,j]))
            splitting_intervals.append(splitting_interval)

    if not splitting:
        return zip(L.active, 
                   pvalues,
                   intervals), L
    else:
        return zip(L.active, 
                   pvalues,
                   intervals,
                   splitting_pvalues,
                   splitting_intervals), L


