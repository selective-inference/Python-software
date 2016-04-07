"""
This module contains a class `lasso`_ that implements
post selection for the lasso
as described in `post selection LASSO`_.


.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238
.. _sample carving: http://arxiv.org/abs/1410.2597

"""

import warnings
from copy import copy

import numpy as np
from scipy.stats import norm as ndist, t as tdist

from regreg.api import (glm, 
                        weighted_l1norm, 
                        simple_problem,
                        coxph)

from ..constraints.affine import (constraints, selection_interval,
                                 interval_constraints,
                                 sample_from_constraints,
                                 gibbs_test,
                                 stack)
from ..distributions.discrete_family import discrete_family


DEBUG = False

def instance(n=100, p=200, s=7, sigma=5, rho=0.3, snr=7,
             random_signs=False, df=np.inf,
             scale=True, center=True):
    """
    A testing instance for the LASSO.
    Design is equi-correlated in the population,
    normalized to have columns of norm 1.

    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.

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

    df : int
        Degrees of freedom for noise (from T distribution).

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
    if center:
        X -= X.mean(0)[None,:]
    if scale:
        X /= (X.std(0)[None,:] * np.sqrt(n))
    beta = np.zeros(p) 
    beta[:s] = snr 
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True

    # noise model

    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df,size=50000))
            return tdist.rvs(df, size=n) / sd_t

    Y = (np.dot(X, beta) + _noise(n, df)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma

class lasso(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \frac{1}{2n} \|y-X\beta\|^2_2 + 
            \lambda \|\beta\|_1

    where $\lambda$ is `lam`.

    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    def __init__(self, loglike, feature_weights):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        feature_weights : np.ndarray
            Feature weights for L-1 penalty. If a float,
            it is brodcast to all features.

        """

        self.loglike = loglike
        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

    def fit(self, **solve_args):
        """
        Fit the lasso using `regreg`.
        This sets the attributes `soln`, `onestep` and
        forms the constraints necessary for post-selection inference
        by calling `form_constraints()`.

        Parameters
        ----------

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        Returns
        -------

        soln : np.float
             Solution to lasso.
             
        """

        penalty = weighted_l1norm(self.feature_weights, lagrange=1.)
        problem = simple_problem(self.loglike, penalty)
        _soln = problem.solve(**solve_args)
        self._soln = _soln
        if not np.all(_soln == 0):
            self.active = np.nonzero(_soln != 0)[0]
            self.active_signs = np.sign(_soln[self.active])
            self._active_soln = _soln[self.active]
            H = self.loglike.hessian(self._soln)[self.active][:,self.active]
            Hinv = np.linalg.inv(H)
            G = self.loglike.gradient(self._soln)[self.active]
            delta = Hinv.dot(G)
            self._onestep = self._active_soln - delta
            self.active_penalized = self.feature_weights[self.active] != 0
            self._constraints = constraints(-np.diag(self.active_signs)[self.active_penalized],
                                             (self.active_signs * delta)[self.active_penalized],
                                             covariance=Hinv)
        else:
            self.active = []
        return self._soln

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def constraints(self):
        """
        Affine constraints for this LASSO problem.
        These are the constraints determined only
        by the active block.
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
            C = self.constraints
            if C is not None:
                one_step = self._onestep
                for i in range(one_step.shape[0]):
                    eta = np.zeros_like(one_step)
                    eta[i] = 1.
                    _interval = C.interval(eta, one_step,
                                           alpha=self.alpha,
                                           UMAU=self.UMAU)
                    self._intervals.append((self.active[i],
                                            _interval[0], _interval[1]))
            self._intervals = np.array(self._intervals, 
                                       np.dtype([('index', np.int),
                                                 ('lower', np.float),
                                                 ('upper', np.float)]))
        return self._intervals

    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
        " for selection."):
        if not hasattr(self, "_pvals"):
            self._pvals = []
            C = self.constraints
            if C is not None:
                one_step = self._onestep
                for i in range(one_step.shape[0]):
                    eta = np.zeros_like(one_step)
                    eta[i] = 1.
                    _pval = C.pivot(eta, one_step)
                    _pval = 2 * min(_pval, 1 - _pval)
                    self._pvals.append((self.active[i], _pval))
        return self._pvals

    @property
    def onesided_pvalues(self, doc="Onesided tests for active variables adjusted " + \
        " for selection."):
        if not hasattr(self, "_onesided_pvals"):
            self._onesided_pvals = []
            C = self.constraints
            if C is not None:
                one_step = self._onestep
                for i in range(one_step.shape[0]):
                    eta = np.zeros_like(one_step)
                    eta[i] = self.active_signs[i]
                    _pval = C.pivot(eta, one_step)
                    self._onesided_pvals.append((self.active[i], _pval))
        return self._onesided_pvals

    @staticmethod
    def gaussian(X, Y, feature_weights, sigma, quadratic=None):
        loglike = glm.gaussian(X, Y, coef=1. / sigma**2, quadratic=quadratic)
        return lasso(loglike, feature_weights)

    @staticmethod
    def logistic(X, successes, feature_weights, trials=None, quadratic=None):
        loglike = glm.logistic(X, successes, trials=trials, quadratic=quadratic)
        return lasso(loglike, feature_weights)

    @staticmethod
    def coxph(X, times, status, feature_weights, quadratic=None):
        loglike = coxph(X, times, status, quadratic=quadratic)
        return lasso(loglike, feature_weights)

    @staticmethod
    def poisson(X, counts, feature_weights, quadratic=None):
        loglike = glm.poisson(X, counts, quadratic=quadratic)
        return lasso(loglike, feature_weights)

    def summary(self, alternative='twosided'):
        """
        Summary table for inference adjusted for selection.

        Parameters
        ----------

        alternative : str
            One of ["twosided","onesided"]

        Returns
        -------

        pval_summary : np.recarray
            Array with one entry per active variable.
            Columns are 'variable', 'pval', 'lasso', 'onestep', 'lower_trunc', 'upper_trunc', 'sd'.

        """

        if alternative not in ['twosided', 'onesided']:
            raise ValueError("alternative must be one of ['twosided', 'onesided']")

        result = []
        C = self.constraints
        if C is not None:
            one_step = self._onestep
            for i in range(one_step.shape[0]):
                eta = np.zeros_like(one_step)
                if self.active_penalized[i]: # use truncated Gaussian
                    eta[i] = self.active_signs[i]
                    _alt = {"onesided":'greater',
                            'twosided':"twosided"}[alternative]
                    _pval = C.pivot(eta, one_step, alternative=_alt)
                    _bounds = np.array(C.bounds(eta, one_step))
                    sd = _bounds[-1]
                    lower_trunc, est, upper_trunc = sorted(_bounds[:3] * self.active_signs[i])
                else: # use regular Gaussian for Wald test
                    sd = np.sqrt(C.covariance[i,i])
                    Z = one_step[i] / sd
                    _pval = 2 * ndist.sf(np.fabs(Z))
                    lower_trunc = -np.inf
                    upper_trunc = np.inf

                result.append((self.active[i],
                               _pval,
                               self._soln[self.active[i]],
                               one_step[i],
                               lower_trunc,
                               upper_trunc,
                               sd))

        return np.array(result,
                        np.dtype([('variable', np.int),
                                  ('pval', np.float),
                                  ('lasso', np.float),
                                  ('onestep', np.float),
                                  ('lower_trunc', np.float),
                                  ('upper_trunc', np.float),
                                  ('sd', np.float)]))

def nominal_intervals(lasso_obj):
    """
    Intervals for OLS parameters of active variables
    that have not been adjusted for selection.
    """
    unadjusted_intervals = []

    if lasso_obj.active is not []:
        SigmaE = lasso_obj.constraints.covariance
        for i in range(lasso_obj.active.shape[0]):
            eta = np.ones_like(lasso_obj._onestep)
            eta[i] = 1.
            center = lasso_obj._onestep[i]
            width = ndist.ppf(1-lasso_obj.alpha/2.) * np.sqrt(SigmaE[i,i])
            _interval = [center-width, center+width]
            unadjusted_intervals.append((lasso_obj.active[i], eta, center,
                                         _interval))
    return unadjusted_intervals

def _constraint_from_data(X_E, X_notE, active_signs, E, lam, sigma, R):

    n, p = X_E.shape[0], X_E.shape[1] + X_notE.shape[1]
    if np.array(lam).shape == ():
        lam = np.ones(p) * lam

    # inactive constraints
    den = np.hstack([lam[~E], lam[~E]])[:,None]
    A0 = np.vstack((R, -R)) / den
    b_tmp = np.dot(X_notE.T, np.dot(np.linalg.pinv(X_E.T), lam[E] * active_signs)) / lam[~E] 
    b0 = np.concatenate((1.-b_tmp, 1.+b_tmp))
    _inactive_constraints = constraints(A0, b0)
    _inactive_constraints.covariance *= sigma**2

    # active constraints
    C = np.linalg.inv(np.dot(X_E.T, X_E))
    A1 = -np.dot(np.diag(active_signs), np.dot(C, X_E.T))
    b1 = -active_signs * np.dot(C, active_signs*lam[E])

    _active_constraints = constraints(A1, b1)
    _active_constraints.covariance *= sigma**2

    _constraints = stack(_active_constraints,
                         _inactive_constraints)
    _constraints.covariance *= sigma**2
    return _active_constraints, _inactive_constraints, _constraints

def standard_lasso(X, y, sigma=1, lam_frac=1.):
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
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 50000)))).max(0)) / sigma
    lasso_selector = lasso.gaussian(X, y, lam, sigma=sigma)
    lasso_selector.fit()
    return lasso_selector

def data_carving(X, y, 
                 lam_frac=2.,
                 sigma=1., 
                 stage_one=None,
                 split_frac=0.9,
                 coverage=0.95, 
                 ndraw=8000,
                 burnin=2000,
                 splitting=False,
                 compute_intervals=True,
                 UMPU=False):

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
        Defaults to 8000.

    burnin : int (optional)
        Defaults to 2000.

    splitting : bool (optional)
        If True, also return splitting pvalues and intervals.

    compute_intervals : bool (optional)
        Compute selective intervals?

    UMPU : bool (optional)
        Perform the UMPU test?
      
    Returns
    -------

    results : [(variable, pvalue, interval)
        Indices of active variables, 
        selected (twosided) pvalue and selective interval.
        If splitting, then each entry also includes
        a (split_pvalue, split_interval) using stage_two
        for inference.

    stage_one : `lasso`
        Results of fitting LASSO to stage one data.

    """

    n, p = X.shape
    first_stage, stage_one, stage_two = split_model(y, X,
                                                    sigma=sigma,
                                                    lam_frac=lam_frac,
                                                    split_frac=split_frac,
                                                    stage_one=stage_one)
    splitn = stage_one.shape[0]

    L = first_stage # shorthand
    s = sparsity = L.active.shape[0]

    if splitn < n:

        # quantities related to models fit on
        # stage_one and full dataset

        y1, X1 = y[stage_one], X[stage_one]
        X_E = X[:,L.active]
        X_Ei = np.linalg.pinv(X_E)
        X_E1 = X1[:,L.active]
        X_Ei1 = np.linalg.pinv(X_E1)

        inv_info_E = np.dot(X_Ei, X_Ei.T)
        inv_info_E1 =np.dot(X_Ei1, X_Ei1.T)

        beta_E = np.dot(X_Ei, y)
        beta_E1 = np.dot(X_Ei1, y[stage_one])

        if n - splitn > s:

            linear_part = np.zeros((s, 2*s))
            linear_part[:, s:] = -np.diag(L.active_signs)
            b = L.constraints.offset
            con = constraints(linear_part, b)

            # specify covariance of 2s Gaussian vector

            cov = np.zeros((2*s, 2*s))
            cov[:s, :s] = inv_info_E
            cov[s:, :s] = inv_info_E
            cov[:s, s:] = inv_info_E
            cov[s:, s:] = inv_info_E1

            con.covariance[:] = cov * sigma**2

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
            linear_part[:, :s] = -np.diag(L.active_signs)
            b = L.constraints.offset
            con = constraints(linear_part, b)

            # specify covariance of Gaussian vector

            cov = np.zeros((s + n - splitn, s + n - splitn))
            cov[:s, :s] = inv_info_E1
            cov[s:, :s] = 0
            cov[:s, s:] = 0
            cov[s:, s:] = np.identity(n - splitn) 

            con.covariance[:] = cov * sigma**2

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
            y2, X2 = y[stage_two], X[stage_two]
            X_E2 = X2[:,L.active]
            X_Ei2 = np.linalg.pinv(X_E2)
            beta_E2 = np.dot(X_Ei2, y2)
            inv_info_E2 = np.dot(X_Ei2, X_Ei2.T)

            splitting_pvalues = []
            splitting_intervals = []

            if n - splitn < s:
                warnings.warn('not enough data for second stage of sample splitting')

            split_cutoff = np.fabs(ndist.ppf((1. - coverage) / 2))

        # compute p-values intervals

        cov_inv = np.linalg.pinv(con.covariance)

        for j in range(X_E.shape[1]):

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
                                             UMPU=UMPU,
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
                                             UMPU=UMPU)
                intervals.append((np.nan, np.nan))

            pval = family.cdf(0, observed)
            pval = 2 * min(pval, 1 - pval)

            pvalues.append(pval)

            if splitting:

                if s < n - splitn: # enough data to generically
                                   # test hypotheses. proceed as usual

                    split_pval = ndist.cdf(beta_E2[j] / (np.sqrt(inv_info_E2[j,j]) * sigma))
                    split_pval = 2 * min(split_pval, 1. - split_pval)
                    splitting_pvalues.append(split_pval)

                    splitting_interval = (beta_E2[j] - 
                                          split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma,
                                          beta_E2[j] + 
                                          split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma)
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
        intervals = np.array([L.intervals['lower'], L.intervals['upper']]).T
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
                sigma=1, 
                lam_frac=1.,
                split_frac=0.9,
                stage_one=None):

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

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$. Defaults to 2.

    split_frac : float (optional)
        What proportion of the data to use in the first stage?
        Defaults to 0.9.

    stage_one : [np.array(np.int), None] (optional)
        Index of data points to be used in  first stage.
        If None, a randomly chosen set of entries is used based on
        `split_frac`.

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

    first_stage = standard_lasso(X1, y1, sigma=sigma, lam_frac=lam_frac)
    return first_stage, stage_one, stage_two

def additive_noise(X, 
                   y, 
                   sigma, 
                   lam_frac=1.,
                   perturb_frac=0.2, 
                   y_star=None,
                   coverage=0.95,
                   ndraw=8000, 
                   compute_intervals=True,
                   burnin=2000):


    """
    
    Additive noise LASSO.

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    lam_frac : float (optional)
        Multiplier for choice of $\lambda$. Defaults to 2.

    perturb_frac : float (optional)
        How much noise to add? Noise added has variance
        proportional to existing variance.

    coverage : float
        Coverage for selective intervals. Defaults to 0.95.

    ndraw : int (optional)
        How many draws to keep from Gibbs hit-and-run sampler.
        Defaults to 8000.

    burnin : int (optional)
        Defaults to 2000.

    compute_intervals : bool (optional)
        Compute selective intervals?
      
    Returns
    -------

    results : [(variable, pvalue, interval)
        Indices of active variables, 
        selected (twosided) pvalue and selective interval.
        If splitting, then each entry also includes
        a (split_pvalue, split_interval) using stage_two
        for inference.

    randomized_lasso : `lasso`
        Results of fitting LASSO to randomized data.

    """

    n, p = X.shape

    # Add some noise to y and fit the LASSO at a fixed lambda

    gamma = np.sqrt(perturb_frac) * sigma 
    sigma_star = np.sqrt(sigma**2 + gamma**2)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 5000)))).max(0)) * sigma_star
    y_star = y + np.random.standard_normal(n) * gamma

    randomized_lasso = L = standard_lasso(X, y_star, sigma=sigma_star, lam_frac=lam_frac)
    L.fit()

    # Form the constraint matrix on (y,y^*)
    X_E = X[:,L.active]
    X_Ei = np.linalg.pinv(X_E)
    Cov_E = np.dot(X_Ei, X_Ei.T)
    W_E = np.dot(Cov_E, L.active_signs)

    pvalues = []
    intervals = []

    beta_E = np.dot(X_Ei, y)

    # compute each pvalue
    for j in range(X_E.shape[1]):
        s_obs = L.active.shape[0]
        keep = np.ones(s_obs, np.bool)
        keep[j] = 0

        # form the 2s Gaussian vector we will condition on

        X_minus_j = X_E[:,keep]
        P_minus_j = np.dot(X_minus_j, np.linalg.pinv(X_minus_j))
        R_minus_j = np.identity(n) - P_minus_j

        theta_E = L.active_signs * (np.dot(X_Ei, np.dot(P_minus_j, y)) - lam * W_E)
        scale = np.sqrt(Cov_E[j,j])
        kappa = 1. / scale**2
        alpha_E = kappa * L.active_signs * Cov_E[j]
        A = np.hstack([-alpha_E.reshape((s_obs,1)), np.identity(s_obs)])
        con = constraints(A, theta_E)
        cov = np.zeros((s_obs+1, s_obs+1))
        cov[0,0] = scale**2 * sigma**2
        cov[1:,1:] = Cov_E * gamma**2 * np.outer(L.active_signs, L.active_signs)
        con.covariance[:] = cov
        initial = np.zeros(s_obs+1)
        initial[0] = beta_E[j]
        initial[1:] = -np.dot(X_Ei, y_star-y) * L.active_signs
        eta = np.zeros(s_obs+1)
        eta[0] = 1.

        observed = (initial * eta).sum()

        if compute_intervals:
            _, _, _, family = gibbs_test(con,
                                         initial,
                                         eta,
                                         UMPU=False,
                                         sigma_known=True,
                                         ndraw=ndraw,
                                         burnin=burnin,
                                         how_often=5,
                                         tilt=np.dot(con.covariance, 
                                                     eta))

            lower_lim, upper_lim = family.equal_tailed_interval(observed, 1 - coverage)

            # in the model we've chosen, the parameter beta is associated
            # to the natural parameter as below
            # exercise: justify this!

            lower_lim_final = np.dot(eta, np.dot(con.covariance, eta)) * lower_lim
            upper_lim_final = np.dot(eta, np.dot(con.covariance, eta)) * upper_lim

            intervals.append((lower_lim_final, upper_lim_final))

        else:
            _, _, _, family = gibbs_test(con,
                                         initial,
                                         eta,
                                         UMPU=False,
                                         sigma_known=True,
                                         ndraw=ndraw,
                                         burnin=burnin,
                                         how_often=5,
                                         tilt=np.dot(con.covariance, 
                                                     eta))

            intervals.append((np.nan, np.nan))

        pval = family.cdf(0, observed)
        pval = 2 * min(pval, 1 - pval)
        pvalues.append(pval)

    return zip(L.active, 
               pvalues,
               intervals), randomized_lasso

