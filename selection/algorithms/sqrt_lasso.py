"""
This module contains a class `sqrt_lasso`_ that implements
post selection for the square root lasso.

Code based on algorithms described in http://arxiv.org/abs/1504.08031.

"""

from copy import copy

import numpy as np, warnings
from scipy.stats import norm as ndist, chi as chidist
from scipy.interpolate import interp1d
from scipy.stats import t as tdist
from statsmodels.api import OLS

# regreg http://github.com/regreg 

import regreg.api as rr

# local

from .lasso import _constraint_from_data
from .sqrt_lasso_objective import solve_sqrt_lasso

from ..constraints.quasi_affine import (constraints_unknown_sigma, 
                                        constraints as quasi_affine,
                                        orthogonal as orthogonal_QA)
from ..constraints.affine import (constraints as affine_constraints, 
                                  gibbs_test,
                                  sample_from_sphere)
from ..truncated import find_root
from ..distributions.discrete_multiparameter import multiparameter_family
from ..distributions.discrete_family import discrete_family


class sqrt_lasso(object):

    r"""
    A class for the square-root LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \|y-X\beta\|^2 + 
             \lambda \|\beta\|_1

    where $\lambda$ is `lam` and 

    .. math::

       \lambda_{\max} = \frac{1}{n} \|X^Ty\|_{\infty}

    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    def __init__(self, y, X, weights, quadratic=None):

        """
        Parameters
        ----------

        y : np.float(y)
            The target, in the model $y = X\beta$

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        weights : np.float(p) or float
            Coefficients in weighted L-1 penalty in
            optimization problem. If a float,
            weights are proportional to 1.

        """
        
        n, p = X.shape

        self.y = y
        self.X = X
        n, p = X.shape

        if np.array(weights).shape == ():
            weights = weights * np.ones(p)
        self.weights = weights
        self.quadratic = quadratic

    def fit(self, **solve_args):
        """
        Fit the square root LASSO using `regreg`
        using `weights=self.weights.`

        Parameters
        ----------

        solve_args : dict
            Arguments passed to regreg solver.

        Returns
        -------

        soln : np.float
             Solution to lasso with `sklearn_alpha=self.lagrange`.

        """

        y, X = self.y, self.X
        n, p = self.X.shape
        self._soln, self._loss = solve_sqrt_lasso(X, y, self.weights, quadratic=self.quadratic, **solve_args)

        beta = self._soln
        
        self.active = (beta != 0)             # E
        self.inactive = ~self.active

        self._subgrad = -(self._loss.smooth_objective(beta, 'grad') + self.quadratic.objective(beta, 'grad'))

        nactive = self.active.sum()           # |E|
        if nactive:
            self.z_E = np.sign(beta[self.active]) # z_E

            # calculate the "partial correlation" operator R = X_{-E}^T (I - P_E)

            X_E = self._X_E = self.X[:,self.active]
            X_notE = self.X[:,~self.active]
            self._XEinv = np.linalg.pinv(X_E)

            self.df_E = n - nactive

            self.P_E = np.dot(X_E, self._XEinv)
            self.R_E = np.identity(n) - self.P_E

            w_E = np.dot(self._XEinv.T, self.weights[self.active] * self.z_E)
            sigma_multiplier = np.sqrt(self.df_E / (1 - np.linalg.norm(w_E)**2))
            self.sigma_E = np.linalg.norm((y - np.dot(self.P_E, y))) / np.sqrt(self.df_E)

            (self._active_constraints, 
             self._inactive_constraints, 
             self._constraints) = _constraint_from_data(X_E,
                                                        X_notE,
                                                        self.z_E,
                                                        self.active, 
                                                        sigma_multiplier * self.sigma_E * self.weights,
                                                        self.sigma_E,
                                                        np.dot(X_notE.T, self.R_E))

            W_E = np.dot(self._XEinv, w_E)
            s_E = np.sign(self.z_E * W_E)
            self._S_trunc_denominator = denominator = sigma_multiplier * W_E * self.z_E

            self.S_trunc_interval = self.compute_sigma_truncation_interval(np.dot(self._XEinv, y))

            # HACK to make things more stable?
            self.S_trunc_interval[0] = 0

            self._quasi_affine_constraints = orthogonal_QA(self._active_constraints.linear_part,
                                                           np.zeros(self._active_constraints.linear_part.shape[0]),
                                                           self._active_constraints.offset / (self.sigma_E * np.sqrt(self.df_E)),
                                                           (self.sigma_E * np.sqrt(self.df_E))**2,
                                                           self.df_E)

            # for metropolis hastings data carving sampler

            self.full_quasi = quasi_affine(self.constraints.linear_part,
                                           np.zeros(self.constraints.linear_part.shape[0]),
                                           self.constraints.offset / (self.sigma_E * np.sqrt(self.df_E)),
                                           self.R_E)

            cov = np.identity(n) * self.sigma_hat**2 
            for con in [self._active_constraints,
                        self._inactive_constraints,
                        self._constraints,
                        self._quasi_affine_constraints]:
                con.covariance[:] = cov

        else:
            self.df_E = self.y.shape[0]
            self.sigma_E = np.linalg.norm(y) / np.sqrt(self.df_E)
            self.S_trunc_interval = [0, np.inf]
            self._active_constraints = self._inactive_constraints = self._constraints = None

        self.active = np.nonzero(self.active)[0]
        self.inactive = np.nonzero(self.inactive)[0]

    def compute_sigma_truncation_interval(self, coef, raise_if_outside=False):
        numerator = coef * self.z_E
        denominator = self._S_trunc_denominator
        s_E = np.sign(self.z_E * denominator)
        S_upper = np.min((numerator / denominator)[denominator > 0])
        if np.any(denominator < 0):
            S_lower = max(np.max((numerator / denominator)[denominator < 0]), 0)
        else:
            S_lower = 0.
        if not (self.sigma_E > S_lower and self.sigma_E < S_upper) and raise_if_outside:
            raise ValueError('obseved sigma_hat not in expected truncation interval')
        return [S_lower, S_upper]

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
    def quasi_affine_constraints(self):
        """
        Quasi-affine constraints imposed on the
        active variables by the KKT conditions.
        """
        return self._quasi_affine_constraints

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
    def sigma_hat(self):
        """
        Estimate of noise in selected model.
        """
        if not hasattr(self, "_sigma_hat"):
            if self.active.shape[0] > 0:
                self._sigma_hat = estimate_sigma(self.sigma_E, 
                                                 self.df_E, 
                                                 self.S_trunc_interval[0],
                                                 self.S_trunc_interval[1])
            else:
                self._sigma_hat = self.sigma_E
        return self._sigma_hat

    @property
    def scaled_lasso_sigma(self):
        """
        Scaled LASSO estimate of sigma
        """
        if not hasattr(self, "_scaled_sigma_hat"):
            if self.active.shape[0] > 0:
                resid = self.y - np.dot(self.X, self.soln)
                self._scaled_sigma_hat = np.sqrt(np.linalg.norm(resid)**2 / self.df_E)
            else:
                self._scaled_sigma_hat = self.sigma_E
        return self._scaled_sigma_hat

    @property
    def intervals(self):
        """
        Intervals for OLS parameters of active variables
        adjusted for selection.
        """
        raise NotImplementedError('intervals are coming soon')

    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
        " for selection."):
        if not hasattr(self, "_pvals"):
            self._pvals = None
            if self.active.shape[0] > 0:
                self._pvals = []
                C = self.active_constraints
                XEinv = self._XEinv
                if XEinv is not None:
                    for i in range(XEinv.shape[0]):
                        eta = XEinv[i]
                        _pval = self.quasi_affine_constraints.pivot(eta, 
                                                                    self.y,
                                                                    alternative='twosided')
                        self._pvals.append((self.active[i], _pval))
        return self._pvals

    @property
    def active_gaussian_pval(self):
        if not hasattr(self, "_gaussian_pvals"):
            self._gaussian_pvals = None
            if self.active.shape[0] > 0:
                self._gaussian_pvals = []
                C = self.active_constraints
                XEinv = self._XEinv
                n, p = self.X.shape
                if XEinv is not None:
                    for i in range(XEinv.shape[0]):
                        eta = XEinv[i]
                        _gaussian_pval = C.pivot(eta, self.y, alternative="twosided")
                        self._gaussian_pvals.append((self.active[i], _gaussian_pval))
                self._gaussian_pvals = np.array(self._gaussian_pvals,
                                                np.dtype([('variable', np.int),
                                                          ('pvalue', np.float)]))
        return self._gaussian_pvals

    @property
    def active_gaussian_intervals(self):
        if not hasattr(self, "_gaussian_intervals"):
            self._gaussian_intervals = None
            if self.active.shape[0] > 0:
                self._gaussian_intervals = []
                C = self.active_constraints
                XEinv = self._XEinv
                n, p = self.X.shape
                if XEinv is not None:
                    for i in range(XEinv.shape[0]):
                        eta = XEinv[i]
                        _interval = C.interval(eta, self.y,
                                               alpha=self.alpha)
                        self._gaussian_intervals.append((self.active[i], _interval[0], _interval[1]))
                self._gaussian_intervals = np.array(self._gaussian_intervals,
                                                    np.dtype([('variable', np.int),
                                                              ('lower', np.float),
                                                              ('upper', np.float)]))
        return self._gaussian_intervals

    def goodness_of_fit(self, statistic, 
                        force=False,
                        alternative='twosided', 
                        ndraw=5000,
                        burnin=2000,
                        ):

        """

        Compute a goodness of fit test based on a given
        statistic applied to 

        .. math::

             U_{-E}(y) = (I-P_E)y / \|(I-P_E)y\|_2

        which is ancillary for the selected model.

        Parameters
        ----------

        statistic : callable
            Statistic to compute on observed $U_{-E}$ as well
            as sample from null distribution.

        alternative : str
            One of ['greater', 'less', 'twosided']. Determines
            how pvalue is computed, based on upper tail, lower tail
            or equal tail.

        force : bool
            Resample from $U_{-E}$ under the null even if
            the instance already has a null sample.

        ndraw : int (optional)
            If a null sample is to be drawn, how large a sample?
            Defaults to 1000.

        burnin : int (optional)
            If a null sample is to be drawn, how long a burnin?
            Defaults to 1000.

        Returns
        -------

        pvalue : np.float
             Two-tailed p-value.

        """

        if not hasattr(self, "_goodness_of_fit_sample") or force:
            if self.active.shape[0] > 0:
                con = self.inactive_constraints
                conditional_con = con.conditional(self._X_E.T, np.dot(self._X_E.T, self.y))

                Z, W = sample_from_sphere(conditional_con, 
                                          self.y,
                                          ndraw=ndraw,
                                          burnin=burnin)  
                U_notE_sample = np.dot(self.R_E, Z.T).T
                U_notE_sample /= np.sqrt((U_notE_sample**2).sum(1))[:,None]
                self._goodness_of_fit_sample = multiparameter_family(U_notE_sample, W)
                self._goodness_of_fit_observed = np.dot(self.R_E, self.y) / np.linalg.norm(np.dot(self.R_E, self.y))

            else:
                n, p = self.X.shape
                U_sample = np.random.standard_normal((ndraw, n))
                U_sample /= np.sqrt((U_sample**2).sum(1))[:, None]
                self._goodness_of_fit_sample = multiparameter_family(U_sample, np.ones(U_sample.shape[0]))
                self._goodness_of_fit_observed = self.y / np.linalg.norm(self.y)

        null_sample = self._goodness_of_fit_sample.sufficient_stat
        importance_weights = self._goodness_of_fit_sample.weights
        null_statistic = np.array([statistic(u) for u in null_sample])
        observed = statistic(self._goodness_of_fit_observed)
        family = discrete_family(null_statistic, importance_weights)

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("expecting alternative to be in ['greater', 'less', 'twosided']")

        if alternative == 'less':
            pvalue = family.cdf(0, observed)
        elif alternative == 'greater':
            pvalue = family.ccdf(0, observed)
        else:
            pvalue = family.cdf(0, observed)
            pvalue = 2 * min(pvalue, 1. - pvalue)

        return pvalue

def nominal_intervals(sqrtL):
    """
    Intervals for OLS parameters of active variables
    that have not been adjusted for selection.

    Notes
    -----

    These intervals do not have any coverage guarantees.
    """
    if not hasattr(sqrtL, "_constraints"):
        sqrtL.form_constraints()
    _intervals_unadjusted = []
    XEinv = sqrtL._XEinv
    SigmaE = sqrtL.sigma_hat**2 * np.dot(XEinv, XEinv.T)
    for i in range(sqrtL.active.shape[0]):
        eta = XEinv[i]
        center = (eta*sqrtL.y).sum()
        width = tdist.ppf(1-sqrtL.alpha/2., sqrtL.df_E) * np.sqrt(SigmaE[i,i])
        _interval = [center-width, center+width]
        _intervals_unadjusted.append((sqrtL.active[i], eta, (eta*sqrtL.y).sum(), 
                                _interval))
    return _intervals_unadjusted

def nominal_pvalues(sqrtL):
    """
    P-values for OLS parameters of active variables
    that have not been adjusted for selection.

    Notes
    -----

    These p-values do not have any selective Type I error control.
    """
    if not hasattr(sqrtL, "_constraints"):
        sqrtL.form_constraints()
    _pvalues_unadjusted = []
    XEinv = sqrtL._XEinv
    SigmaE = sqrtL.sigma_hat**2 * np.dot(XEinv, XEinv.T)
    for i in range(sqrtL.active.shape[0]):
        eta = XEinv[i]
        center = (eta*sqrtL.y).sum()
        T = center / np.sqrt(SigmaE[i,i])
        _pval = tdist.cdf(T, sqrtL.df_E)
        _pval = 2 * min(_pval, 1 - _pval)
        _pvalues_unadjusted.append((sqrtL.active[i], T, _pval))
    return _pvalues_unadjusted

def estimate_sigma(observed, truncated_df, lower_bound, upper_bound, untruncated_df=0, factor=3, npts=50, nsample=2000):
    """

    Produce an estimate of $\sigma$ from a constrained
    error sum of squares. The relevant distribution is a
    scaled $\chi^2$ restricted to $[0,U]$ where $U$ is `upper_bound`.

    Parameters
    ----------

    observed : float
        The observed sum of squares.

    truncated_df : float
        Degrees of freedom of the truncated $\chi^2$ in the sum of squares.
        The observed sum is assumed to be the sum
        of an independent untruncated $\chi^2$ and the truncated one.

    lower_bound : float
        Lower limit of truncation interval.
    
    upper_bound : float
        Upper limit of truncation interval.
    
    untruncated_df : float
        Degrees of freedom of the untruncated $\chi^2$ in the sum of squares.

    factor : float
        Range of candidate values is 
        [observed/factor, observed*factor]

    npts : int
        How many candidate values for interpolator.

    nsample : int
        How many samples for each expected value
        of the truncated sum of squares.

    Returns
    -------

    sigma_hat : np.float
         Estimate of $\sigma$.
    
    """

    if untruncated_df < 5:
        linear_term = truncated_df**(0.5)
    else:
        linear_term = 0

    total_df = untruncated_df + truncated_df

    values = np.linspace(1./factor, factor, npts) * observed
    expected = 0 * values
    for i, value in enumerate(values):
        P_upper = chidist.cdf(upper_bound * np.sqrt(truncated_df) / value, truncated_df) 
        P_lower = chidist.cdf(lower_bound * np.sqrt(truncated_df) / value, truncated_df) 
        U = np.random.sample(nsample)
        if untruncated_df > 0:
            sample = (chidist.ppf((P_upper - P_lower) * U + P_lower, truncated_df)**2 + chidist.rvs(untruncated_df, size=nsample)**2) * value**2
        else:
            sample = (chidist.ppf((P_upper - P_lower) * U + P_lower, truncated_df) * value)**2
        expected[i] = np.mean(sample) 

        if expected[i] >= 1.5 * (observed**2 * total_df + observed**2 * linear_term):
            break

    interpolant = interp1d(values, expected + values**2 * linear_term)
    V = np.linspace(1./factor,factor,10*npts) * observed

    # this solves for the solution to 
    # expected(sigma) + sqrt(df) * sigma^2 = observed SS * (1 + sqrt(df))
    # the usual "MAP" estimator would have RHS just observed SS
    # but this factor seems to ``correct it''.
    # it is such that if there were no selection it would be 
    # the usual unbiased estimate

    sigma_hat = np.min(V[interpolant(V) >= observed**2 * total_df + observed**2 * linear_term])

    return sigma_hat

def choose_lambda(X, quantile=0.95, ndraw=10000):
    """
    Choose a value of `lam` for the square-root LASSO
    based on an empirical quantile of the distribution of

    $$
    \frac{\|X^T\epsilon\|_{\infty}}{\|\epsilon\|_2}.
    $$

    Parameters
    ----------

    X : np.float((n, p))
        Design matrix.

    quantile : float
        What quantile should we use?

    ndraw : int
        How many draws?

    """

    X = rr.astransform(X)
    n, p = X.output_shape[0], X.input_shape[0]
    E = np.random.standard_normal((n, ndraw))
    E /= np.sqrt(np.sum(E**2, 0))[None,:]
    return np.percentile(np.fabs(X.adjoint_map(E)).max(0), 100*quantile)

def data_carving(y, X, 
                 lam_frac=1.,
                 stage_one=None,
                 split_frac=0.9,
                 coverage=0.95, 
                 ndraw=5000,
                 burnin=1000,
                 splitting=False,
                 compute_intervals=False,
                 fit_args={}):

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

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$. Defaults to 2.

    coverage : float
        Coverage for selective intervals. Defaults to 0.95.

    stage_one : [np.array(np.int), None] (optional)
        Index of data points to be used in  first stage.
        If None, a randomly chosen set of entries is used based on
        `split_frac`.

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

    fit_args : dict (optional)
        Arguments passed to `sqrt_lasso.fit`

    Returns
    -------

    results : [(variable, pvalue, interval)]
        Indices of active variables, 
        selected (twosided) pvalue and selective interval.
        If splitting, then each entry also includes
        a (split_pvalue, split_interval) using stage_two
        for inference.

    L : sqrt_lasso
        Instance of class `sqrt_lasso` that was fit
        to first stage data.
    """

    n, p = X.shape
    first_stage, stage_one, stage_two = split_model(y, X,
                                                    lam_frac=lam_frac,
                                                    split_frac=split_frac,
                                                    stage_one=stage_one,
                                                    fit_args=fit_args)
    n1 = splitn = stage_one.shape[0]
    n2 = n - n1

    L = first_stage # shorthand

    # quantities related to models fit on
    # stage_one and full dataset

    if n1 < n:
        y1, X1 = y[stage_one], X[stage_one]
        X_E = X[:,L.active]
        X_Ei = np.linalg.pinv(X_E)
        X_E1 = X1[:,L.active]
        X_Ei1 = np.linalg.pinv(X_E1)
        R1 = np.identity(n1) - np.dot(X_E1, X_Ei1)
        selector1 = np.identity(n)[stage_one]
        R_stageone = np.dot(selector1.T, np.dot(R1, selector1))

        inv_info_E = np.dot(X_Ei, X_Ei.T)
        inv_info_E1 = np.dot(X_Ei1, X_Ei1.T)

        s = sparsity = L.active.shape[0]
        beta_E = np.dot(X_Ei, y)
        beta_E1 = np.dot(X_Ei1, y[stage_one])
        sigma_E1 = np.linalg.norm(y[stage_one] - np.dot(X_E1, beta_E1)) / np.sqrt(stage_one.sum() - L.active.shape[0])
        sigma_E = np.linalg.norm(y - np.dot(X_E, beta_E)) / np.sqrt(n - L.active.shape[0])

        sigma_hat = L.sigma_hat 

        if n2 > s or splitting:

            y2, X2 = y[stage_two], X[stage_two]
            X_E2 = X2[:,L.active]
            X_Ei2 = np.linalg.pinv(X_E2)

        if n2 > s:
            RSS_2 = np.linalg.norm(y2 - np.dot(X_E2, np.dot(X_Ei2, y2)))**2
            sigma_hat = np.sqrt(((n1-s)*sigma_hat**2 + RSS_2) / (n1 + n2 - 2*s))
            linear_part = np.zeros((s, 2*s))
            linear_part[:, s:] = -np.diag(L.z_E)
            b = L.active_constraints.offset
            con = affine_constraints(linear_part, b)

            # specify covariance of 2s Gaussian vector

            cov = np.zeros((2*s, 2*s))
            cov[:s, :s] = inv_info_E
            cov[s:, :s] = inv_info_E
            cov[:s, s:] = inv_info_E
            cov[s:, s:] = inv_info_E1

            con.covariance[:] = cov * sigma_hat**2

            # for the conditional law
            # we will change the linear function for each coefficient

            selector = np.zeros((s, 2*s))
            selector[:, :s]  = np.identity(s)
            conditional_linear = np.dot(np.dot(X_E.T, X_E), selector) 

            # a valid initial condition

            initial = np.hstack([beta_E, beta_E1]) 
            OLS_func = selector

        else:

            linear_part = np.zeros((s, s + n - splitn))
            linear_part[:, :s] = -np.diag(L.z_E)
            b = L.active_constraints.offset
            con = affine_constraints(linear_part, b)

            # specify covariance of Gaussian vector

            cov = np.zeros((s + n - splitn, s + n - splitn))
            cov[:s, :s] = inv_info_E1
            cov[s:, :s] = 0
            cov[:s, s:] = 0
            cov[s:, s:] = np.identity(n - splitn) 

            con.covariance[:] = cov * sigma_hat**2

            conditional_linear = np.zeros((s, s + n - splitn))
            conditional_linear[:, :s]  = np.linalg.pinv(inv_info_E1)
            conditional_linear[:, s:] = X[stage_two,:][:,L.active].T

            selector1 = np.zeros((s, s + n - splitn))
            selector1[:, :s]  = np.identity(s)
            selector2 = np.zeros((n - splitn, s + n - splitn))
            selector2[:, s:]  = np.identity(n - splitn)

            # write the OLS estimates of full model in terms of X_E1^{dagger}y_1, y2

            OLS_func = np.dot(inv_info_E, conditional_linear) 

            # a valid initial condition

            initial = np.hstack([beta_E1, y[stage_two]]) 

        pvalues = []
        intervals = []

        if splitting:
            if n2 < s:
                warnings.warn('not enough data for second stage of sample splitting')

            y2, X2 = y[stage_two], X[stage_two]
            X_E2 = X2[:,L.active]
            X_Ei2 = np.linalg.pinv(X_E2)
            beta_E2 = np.dot(X_Ei2, y2)
            sigma_E2 = np.linalg.norm(y2 - np.dot(X_E2, beta_E2)) / np.sqrt(n2 - s)

            inv_info_E2 = np.dot(X_Ei2, X_Ei2.T) 

            splitting_pvalues = []
            splitting_intervals = []

            split_cutoff = np.fabs(tdist.ppf((1. - coverage) / 2, n2 - s))

        # compute p-values and (TODO: intervals)

        full_RSS = sigma_E**2 * (n - s)

        do_MH = False
        for j in range(X_E.shape[1]):
            
            if not do_MH: # do Gibbs instead

                keep = np.ones(s, np.bool)
                keep[j] = 0

                eta = OLS_func[j]

                con_cp = copy(con)
                conditional_law = con_cp.conditional(conditional_linear[keep], \
                                                         np.dot(X_E.T, y)[keep])

                # tilt so that samples are closer to observed values
                # the multiplier should be the pseudoMLE so that
                # the observed value is likely 

                observed = (initial * eta).sum()

                if compute_intervals:
                    _, _, _, family = gibbs_test(conditional_law,
                                                 initial, 
                                                 eta,
                                                 sigma_known=True,
                                                 white=False,
                                                 ndraw=ndraw,
                                                 burnin=burnin,
                                                 how_often=10,
                                                 UMPU=False,
                                                 tilt=np.dot(conditional_law.covariance, 
                                                             eta))

                    lower_lim, upper_lim = family.equal_tailed_interval(observed, 1 - coverage)

                    # in the model we've chosen, the parameter beta is associated
                    # to the natural parameter as below
                    # exercise: justify this!

                    lower_lim_final = np.dot(eta, np.dot(conditional_law.covariance, eta)) * lower_lim
                    upper_lim_final = np.dot(eta, np.dot(conditional_law.covariance, eta)) * upper_lim

                    intervals.append((lower_lim_final, upper_lim_final))
                else: # we do not really need to tilt just for p-values
                    _, _, _, family = gibbs_test(conditional_law,
                                                 initial, 
                                                 eta,
                                                 sigma_known=True,
                                                 white=False,
                                                 ndraw=ndraw,
                                                 burnin=burnin,
                                                 how_often=10,
                                                 UMPU=False)
                    intervals.append((np.nan, np.nan))

                pval = family.cdf(0, observed)
                pval = 2 * min(pval, 1 - pval)

                pvalues.append(pval)

            else:
                pval = _MH_sample_data_carve(y, X,
                                             stage_one,
                                             L,
                                             j,
                                             ndraw=ndraw,
                                             burnin=burnin)
                pvalues.append(pval)
                # intervals are not implemented yet
                intervals.append((np.nan, np.nan))

            if splitting:
                if s < n2: # enough data to generically
                                   # test hypotheses. proceed as usual

                    T = beta_E2[j] / (sigma_E2 * np.sqrt(inv_info_E2[j,j]))
                    split_pval = tdist.cdf(T, n2 - s)
                    split_pval = 2 * min(split_pval, 1. - split_pval)
                    splitting_pvalues.append(split_pval)

                    splitting_interval = (beta_E2[j] - 
                                          split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma_E2,
                                          beta_E2[j] + 
                                          split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma_E2)
                    splitting_intervals.append(splitting_interval)
                else:
                    splitting_pvalues.append(np.random.sample())
                    splitting_intervals.append((np.nan, np.nan))

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
    else:
        pvalues = [p for _, p in L.active_pvalues]
        intervals = [o[-1] for o in L.active_gaussian_intervals]
        if splitting:
            splitting_pvalues = np.random.sample(len(pvalues))
            splitting_intervals = [(np.nan, np.nan) for _ in 
                                   range(len(pvalues))]

            return zip(L.active, 
                       pvalues, 
                       intervals,
                       splitting_pvalues,
                       splitting_intervals), L
        else:
            return zip(L.active, 
                       pvalues,
                       intervals), L

def split_model(y, X, 
                lam_frac=1.,
                split_frac=0.9,
                quantile=0.95,
                stage_one=None,
                fit_args={}):

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

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$. Defaults to 2.

    quantile : float (optional)
        Quantile given to `choose_lambda`

    split_frac : float (optional)
        What proportion of the data to use in the first stage?
        Defaults to 0.9.

    stage_one : [np.array(np.int), None] (optional)
        Index of data points to be used in  first stage.
        If None, a randomly chosen set of entries is used based on
        `split_frac`.

    fit_args : dict (optional)
        Arguments passed to `sqrt_lasso.fit`

    Returns
    -------

    first_stage : `lasso`
        Lasso object from stage one.

    stage_one : np.array(int)
        Indices used for stage one.

    stage_two : np.array(int)
        Indices used for stage two.

    """

    n, p = X.shape
    if stage_one is None:
        splitn = int(n*split_frac)
        indices = np.arange(n)
        np.random.shuffle(indices)
        stage_one = indices[:splitn]
        stage_two = indices[splitn:]
    else:
        stage_two = [i for i in np.arange(n) if i not in stage_one]
    y1, X1 = y[stage_one], X[stage_one]

    first_stage = standard_sqrt_lasso(y1, X1, lam_frac=lam_frac, quantile=quantile, fit_args=fit_args)
    return first_stage, stage_one, stage_two

def standard_sqrt_lasso(y, X, lam_frac=1., quantile=0.95, fit_args={}):
    """
    Fit a sqrt-LASSO with a default choice of Lagrange parameter
    equal to `lam_frac` times $\sigma \cdot E(|X^T\epsilon|) / \|\epsilon\|_2$
    with $\epsilon$ IID N(0,1).

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$

    quantile : float (optional)
        Quantile given to `choose_lambda`

    fit_args : dict (optional)
        Arguments passed to `sqrt_lasso.fit`

    Returns
    -------

    lasso_selection : `lasso`
         Instance of `lasso` after fitting. 

    """
    n, p = X.shape

    lam = lam_frac * choose_lambda(X, quantile=quantile)

    sqrtL = sqrt_lasso(y, X, lam)
    sqrtL.fit(**fit_args)
    if sqrtL.active_constraints is not None and not sqrtL.active_constraints(y):
        raise ValueError('y does not satisfy KKT conditions determined by variables and signs-- try increasing min_its or tol in fit_args')
    return sqrtL

def _MH_sample_data_carve(y, X,
                          stage_one,
                          stage_one_L,
                          which_var,
                          ndraw=10000,
                          burnin=1000):


    L, j = stage_one_L, which_var # shorthand

    quasi_constraints = L.full_quasi
    active = np.zeros(X.shape[1], np.bool)
    active[L.active] = True
    active[j] = False

    # find appropriate residual projector
    # could be done a little more efficiently
    # if all variables done at once?

    X_Ej = X[:,active]
    _XEinv = np.linalg.pinv(X_Ej)
    P_Ej = np.dot(X_Ej, _XEinv)
    n = y.shape[0]
    R_Ej = np.identity(n) - P_Ej
    eta = np.dot(R_Ej, X[:,j])
    P_Ej_y = np.dot(P_Ej, y)

    norm_RE_y = np.linalg.norm(np.dot(R_Ej, y))

    def _proposal_step(y_null):
        Z = np.random.standard_normal(y_null.shape)
        R_y = np.dot(R_Ej, y_null)
        Z0 = np.dot(R_Ej, Z)
        direction = Z0 - (Z0*R_y).sum() / (R_y**2).sum() * R_y
        direction /= np.linalg.norm(direction)
        n = y_null.shape[0]
        angle = np.random.beta(1, n/4) * np.pi 
        jump = False
        if np.random.sample() < 0.1:
            jump = True
            angle = np.random.sample() * np.pi * 2
        proposal = np.cos(angle) * R_y + np.sin(angle) * direction * norm_RE_y + P_Ej_y
        test = quasi_constraints(proposal[stage_one])
        if test:
            return True, proposal.copy()
        else:
            return False, y_null.copy()

    eta_sample = []
    y_null = np.dot(R_Ej, y) + P_Ej_y

    acceptance = []
    for i in range(ndraw + burnin):
        accept, y_null = _proposal_step(y_null)
        acceptance.append(accept)
        if i >= burnin:
            eta_sample.append(np.dot(eta, y_null))
    family = discrete_family(eta_sample, np.ones_like(eta_sample))
    pval = family.cdf(0, (eta*y).sum())
    pval = 2 * min(pval, 1 - pval)

    return pval
