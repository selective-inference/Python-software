"""
This module contains a class `sqrt_lasso`_ that implements
post selection for the square root lasso.

"""

import numpy as np, warnings
from scipy.stats import norm as ndist, chi as chidist
from scipy.interpolate import interp1d
from scipy.stats import t as tdist

# regreg http://github.com/regreg 

import regreg.api as rr

# local

from .lasso import _constraint_from_data
from .truncated.T import truncated_T
from .affine import constraints_unknown_sigma, constraints as gaussian_constraints
from .truncated import find_root
from .sample_truncT import sample_truncated_T

class sqlasso_objective(rr.smooth_atom):
    """

    The square-root LASSO objective. Essentially
    smooth, but singular on 
    $\{\beta: y=X\beta\}$.

    This singularity is ignored in solving the problem.
    It might be a problem sometimes?

    """

    _sqrt2 = np.sqrt(2) # often used constant

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self._sqerror = rr.squared_error(X, Y)

    def smooth_objective(self, x, mode='both', check_feasibility=False):

        f, g = self._sqerror.smooth_objective(x, mode='both', check_feasibility=check_feasibility)
        f = self._sqrt2 * np.sqrt(f)
        if mode == 'both':
            return f, g / f
        elif mode == 'grad':
            return g / f
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")

def solve_sqrt_lasso(X, Y, weights=None, initial=None, **solve_kwargs):
    """

    Solve the square-root LASSO optimization problem:

    $$
    \text{minimize}_{\beta} \|y-X\beta\|_2 + D |\beta|,
    $$
    where $D$ is the diagonal matrix with weights on its diagonal.

    Parameters
    ----------

    y : np.float((n,))
        The target, in the model $y = X\beta$

    X : np.float((n, p))
        The data, in the model $y = X\beta$

    weights : np.float
        Coefficients of the L-1 penalty in
        optimization problem, note that different
        coordinates can have different coefficients.

    initial : np.float(p)
        Initial point for optimization.

    solve_kwargs : dict
        Arguments passed to regreg solver.

    """
    X = rr.astransform(X)
    n, p = X.output_shape[0], X.input_shape[0]
    if weights is None:
        lam = choose_lambda(X)
        weights = lam * np.ones((p,))
    loss = sqlasso_objective(X, Y)
    penalty = rr.weighted_l1norm(weights, lagrange=1.)
    problem = rr.simple_problem(loss, penalty)
    if initial is not None:
        problem.coefs[:] = initial
    soln = problem.solve(**solve_kwargs)
    return soln

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

    def __init__(self, y, X, weights):

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

        if np.array(weights).shape == ():
            weights = weights * np.ones(p)
        self.y = y
        self.X = X
        n, p = X.shape
        self.weights = weights

    def fit(self, **solve_kwargs):
        """
        Fit the square root LASSO using `regreg`
        using `weights=self.weights.`

        Parameters
        ----------

        solve_kwargs : dict
            Arguments passed to regreg solver.

        Returns
        -------

        soln : np.float
             Solution to lasso with `sklearn_alpha=self.lagrange`.

        """

        y, X = self.y, self.X
        n, p = self.X.shape
        self._soln = solve_sqrt_lasso(X, y, self.weights, **solve_kwargs)

        beta = self._soln

        self.active = (beta != 0)             # E
        nactive = self.active.sum()           # |E|
        if nactive:
            self.z_E = np.sign(beta[self.active]) # z_E

            # calculate the "partial correlation" operator R = X_{-E}^T (I - P_E)

            X_E = self.X[:,self.active]
            X_notE = self.X[:,~self.active]
            self._XEinv = np.linalg.pinv(X_E)
            self.w_E = np.dot(self._XEinv.T, self.weights[self.active] * self.z_E)
            self.W_E = np.dot(self._XEinv, self.w_E)
            self.s_E = np.sign(self.z_E * self.W_E)

            self.df_E = n - nactive

            self.P_E = np.dot(X_E, self._XEinv)
            self.R_E = np.identity(n) - self.P_E

            _denE = np.sqrt(1 - np.linalg.norm(self.w_E)**2)
            self._c_E = np.linalg.norm(y - np.dot(self.P_E, y)) / _denE

            _covE = np.dot(self._XEinv, self._XEinv.T)
            _diagE = np.sqrt(np.diag(_covE))
            _corE = _covE / np.outer(_diagE, _diagE)
            self.sigma_E = np.linalg.norm((y - np.dot(self.P_E, y))) / np.sqrt(self.df_E)

            (self._active_constraints, 
             self._inactive_constraints, 
             self._constraints) = _constraint_from_data(X_E,
                                                        X_notE,
                                                        self.z_E,
                                                        self.active, 
                                                        self._c_E * self.weights,
                                                        self.sigma_E,
                                                        np.dot(X_notE.T, self.R_E))

            self.U_E = np.dot(self._XEinv, y) / _diagE
            self.T_E = self.U_E / self.sigma_E

            _fracE = np.sqrt(self.df_E) / (_denE * _diagE)
            RHS = _fracE * np.fabs(self.W_E)
            self.alpha_E = self.s_E * RHS / np.sqrt(self.df_E)
            self.S_trunc_interval = np.min((np.fabs(self.U_E) / RHS)[self.s_E == 1])

            cov = np.identity(n) * self.sigma_hat**2 
            for con in [self._active_constraints,
                        self._inactive_constraints,
                        self._constraints]:
                con.covariance[:] = cov

        else:
            self.df_E = self.y.shape[0]
            self.sigma_E = np.linalg.norm(y) / np.sqrt(self.df_E)
            self.S_trunc_interval = np.inf
            self._active_constraints = self._inactive_constraints = self._constraints = None

        self.active = np.nonzero(self.active)[0]

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
    def sigma_hat(self):
        """
        Estimate of noise in selected model.
        """
        if not hasattr(self, "_sigma_hat"):
            if self.active.shape[0] > 0:
                self._sigma_hat = estimate_sigma(self.sigma_E, 
                                                 self.df_E, 
                                                 self.S_trunc_interval)
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
                        (intervals,
                         Tobs) = constraints_unknown_sigma( \
                            C.linear_part,
                            C.offset / self.sigma_E,
                            self.y,
                            eta,
                            self.R_E)
                        truncT = truncated_T(np.array([(interval.lower_value,
                                                        interval.upper_value) for interval in intervals]), self.df_E)
                        sf = truncT.sf(Tobs)
                        if (truncT.intervals.shape == ((1,2)) and np.all(truncT.intervals == [[-np.inf, np.inf]])):
                            raise ValueError('should be truncated')

                        _pval = float(2 * min(sf, 1.-sf))
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
                        if _gaussian_pval < 1e-10:
                            print self.sigma_hat, C.bounds(eta, self.y) 
                        self._gaussian_pvals.append((self.active[i], _gaussian_pval))
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
                        self._gaussian_intervals.append((self.active[i], _interval))
        return self._gaussian_intervals


def estimate_sigma(observed, df, upper_bound, factor=3, npts=50, nsample=2000):
    """

    Produce an estimate of $\sigma$ from a constrained
    error sum of squares. The relevant distribution is a
    scaled $\chi^2$ restricted to $[0,U]$ where $U$ is `upper_bound`.

    Parameters
    ----------

    observed : float
        The observed sum of squares.

    df : float
        Degrees of freedom of the sum of squares.

    upper_bound : float
        Upper limit of truncation interval.
    
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

    values = np.linspace(1./factor, factor, npts) * observed
    expected = 0 * values
    for i, value in enumerate(values):
        P_upper = chidist.cdf(upper_bound * np.sqrt(df) / value, df) 
        U = np.random.sample(nsample)
        sample = chidist.ppf(P_upper * U, df) * value
        expected[i] = np.mean(sample**2) 

        if expected[i] >= 1.1 * (observed**2 * df + observed**2 * df**(0.5)):
            break

    interpolant = interp1d(values, expected + df**(0.5) * values**2)
    V = np.linspace(1./factor,factor,10*npts) * observed
    # this solves for the solution to 
    # expected(sigma) + sqrt(df) * sigma^2 = observed SS * (1 + sqrt(df))
    # the usual "MAP" estimator would have RHS just observed SS
    # but this factor seems to ``correct it''.
    # it is such that if there were no selection it would be 
    # the usual unbiased estimate
    sigma_hat = np.min(V[interpolant(V) >= observed**2 * df + observed**2 * df**(0.5)])
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

    n, p = X.shape
    E = np.random.standard_normal((n, ndraw))
    E /= np.sqrt(np.sum(E**2, 0))[None,:]
    return np.percentile(np.fabs(np.dot(X.T, E)).max(0), 100*quantile)

def data_carving(y, X, 
                 lam_frac=2.,
                 stage_one=None,
                 split_frac=0.9,
                 coverage=0.95, 
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

    Returns
    -------

    results : [(variable, pvalue, interval)
        Indices of active variables, 
        selected (twosided) pvalue and selective interval.
        If splitting, then each entry also includes
        a (split_pvalue, split_interval) using stage_two
        for inference.

    """

    n, p = X.shape
    first_stage, stage_one, stage_two = split_model(y, X,
                                                    lam_frac=lam_frac,
                                                    split_frac=split_frac,
                                                    stage_one=stage_one)
    splitn = stage_one.shape[0]

    L = first_stage # shorthand

    # quantities related to models fit on
    # stage_one and full dataset

    if splitn < n:
        y1, X1 = y[stage_one], X[stage_one]
        X_E = X[:,L.active]
        X_Ei = np.linalg.pinv(X_E)
        X_E1 = X1[:,L.active]
        X_Ei1 = np.linalg.pinv(X_E1)
        R1 = np.identity(splitn) - np.dot(X_E1, X_Ei1)
        selector1 = np.identity(n)[stage_one]
        R_stageone = np.dot(selector1.T, np.dot(R1, selector1))

        info_E = np.dot(X_Ei, X_Ei.T)
        info_E1 = np.dot(X_Ei1, X_Ei1.T)

        s = sparsity = L.active.shape[0]
        beta_E = np.dot(X_Ei, y)
        beta_E1 = np.dot(X_Ei1, y[stage_one])
        sigma_E1 = np.linalg.norm(y[stage_one] - np.dot(X_E1, beta_E1)) / np.sqrt(stage_one.sum() - L.active.shape[0])
        sigma_E = np.linalg.norm(y - np.dot(X_E, beta_E)) / np.sqrt(n - L.active.shape[0])

        if n - splitn > s:

            linear_part = np.zeros((s, 2*s))
            linear_part[:, s:] = -np.diag(L.z_E)
            b = L.active_constraints.offset
            con = gaussian_constraints(linear_part, b)

            # specify covariance of 2s Gaussian vector

            cov = np.zeros((2*s, 2*s))
            cov[:s, :s] = info_E 
            cov[s:, :s] = info_E
            cov[:s, s:] = info_E
            cov[s:, s:] = info_E1

            con.covariance[:] = cov * sigma_E**2

            # for the conditional law
            # we will change the linear function for each coefficient

            selector = np.zeros((s, 2*s))
            selector[:, :s]  = np.identity(s)
            conditional_linear = np.dot(np.linalg.pinv(info_E), selector) 

            # a valid initial condition

            initial = np.hstack([beta_E, beta_E1]) / sigma_E
            OLS_func = selector

        else:

            linear_part = np.zeros((s, s + n - splitn))
            linear_part[:, :s] = -np.diag(L.z_E)
            b = L.active_constraints.offset
            con = gaussian_constraints(linear_part, b)

            # specify covariance of Gaussian vector

            cov = np.zeros((s + n - splitn, s + n - splitn))
            cov[:s, :s] = info_E1
            cov[s:, :s] = 0
            cov[:s, s:] = 0
            cov[s:, s:] = np.identity(n - splitn) 

            con.covariance[:] = cov * sigma_E**2

            conditional_linear = np.zeros((s, s + n - splitn))
            conditional_linear[:, :s]  = np.linalg.pinv(info_E1) 
            conditional_linear[:, s:] = X[stage_two,:][:,L.active].T

            selector1 = np.zeros((s, s + n - splitn))
            selector1[:, :s]  = np.identity(s)
            selector2 = np.zeros((n - splitn, s + n - splitn))
            selector2[:, s:]  = np.identity(n - splitn)

            # write the OLS estimates of full model in terms of X_E1^{dagger}y_1, y2

            OLS_func = np.dot(info_E, conditional_linear)

            # a valid initial condition

            initial = np.hstack([beta_E1, y[stage_two]]) / sigma_E

        DEBUG = True
        if DEBUG:
            print con(initial * sigma_E), 'working'
            print L.active_constraints(y[stage_one]), 'huh'
        pvalues = []
        intervals = []

        if splitting:
            if n - splitn < s:
                warnings.warn('not enough data for second stage of sample splitting')

            y2, X2 = y[stage_two], X[stage_two]
            X_E2 = X2[:,L.active]
            X_Ei2 = np.linalg.pinv(X_E2)
            beta_E2 = np.dot(X_Ei2, y2)
            sigma_E2 = np.linalg.norm(y[stage_two] - np.dot(X_E2, beta_E2)) / np.sqrt(n - splitn - s)

            info_E2 = np.dot(X_Ei2, X_Ei2.T) 

            splitting_pvalues = []
            splitting_intervals = []

            split_cutoff = np.fabs(tdist.ppf((1. - coverage) / 2, n - splitn - s))

        # compute p-values and (TODO: intervals)

        for j in range(X_E.shape[1]):

            keep = np.ones(s, np.bool)
            keep[j] = 0

            eta = OLS_func[j]

            conditional_con = con.conditional(conditional_linear[keep],
                                              np.dot(X_E.T, y)[keep])

            noncentral_param = conditional_con.mean.copy()
            conditional_con.mean *= 0

            #
            # before conditioning:
            #
            # sigma_E1 has splitn - s degrees of freedom : R_stageone
            # numerator s + min(n - splitn, s) degrees of freedom
            #
            # after conditioning:
            #
            # sigma_E1 has splitn - s degrees of freedom : R_stageone
            # P_minus has s-1 degrees of freedom
            # numerator min(n - splitn, s) + 1 having lost s-1 to P_minus
            #
            # therefore, when n - splitn > s we seem to waste data for
            # estimating variance?

            inverse_map, forward_map, white = conditional_con.whiten()
            y_f = forward_map(initial)
            eta_f = forward_map(eta)
            noncentral_param_f = forward_map(noncentral_param)
            T_obs = (eta_f * (noncentral_param_f + y_f)).sum()

            white_samples = sample_truncated_T(white.linear_part,
                                               white.offset,
                                               y_f, 
                                               noncentral_param_f,
                                               splitn - s,
                                               eta_f,
                                               how_often=3,
                                               ndraw=ndraw, 
                                               burnin=burnin)
            T_sample = np.dot(white_samples, eta_f)
            family = discrete_family(T_sample, np.ones_like(T_sample))
            pval = 2 * min(family.cdf(0, T_obs))

            pvalues.append(pval)

            # intervals are still not implemented yet
            intervals.append((np.nan, np.nan))

            if splitting:
                if s < n - splitn: # enough data to generically
                                   # test hypotheses. proceed as usual

                    T = beta_E2[j] / (sigma_E2 * info_E2[j,j])
                    split_pval = tdist.cdf(T, n - splitn - s)
                    split_pval = 2 * min(split_pval, 1. - split_pval)
                    splitting_pvalues.append(split_pval)

                    splitting_interval = (beta_E2[j] - 
                                          split_cutoff * np.sqrt(info_E2[j,j]) * sigma_E2,
                                          beta_E2[j] + 
                                          split_cutoff * np.sqrt(info_E2[j,j]) * sigma_E2)
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

    first_stage = standard_sqrt_lasso(y1, X1, lam_frac=lam_frac, quantile=quantile)
    return first_stage, stage_one, stage_two

def standard_sqrt_lasso(y, X, lam_frac=1., quantile=0.95):
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

    Returns
    -------

    lasso_selection : `lasso`
         Instance of `lasso` after fitting. 

    """
    n, p = X.shape

    lam = lam_frac * choose_lambda(X, quantile=quantile)

    sqrtL = sqrt_lasso(y, X, lam)
    sqrtL.fit()
    return sqrtL
