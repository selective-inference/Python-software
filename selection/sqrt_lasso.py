"""
This module contains a class `sqrt_lasso`_ that implements
post selection for the square root lasso.

"""

import numpy as np
from scipy.stats import norm as ndist, chi as chidist
from scipy.interpolate import interp1d
from mpmath import betainc, mp
mp.dps = 150

# regreg http://github.com/regreg 

import regreg.api as rr

# local

from .lasso import _constraint_from_data
from .truncated import T as truncated_T
from .affine import constraints_unknown_sigma, constraints as gaussian_constraints
from .truncated import find_root

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

def solve_sqrt_lasso(X, Y, weights=None, **solve_kwargs):
    """

    Solve the square-root LASSO optimization problem:

    $$
    \text{minimize}_{\beta} \|y-X\beta\|_2 + D |\beta|,
    $$
    where $D$ is the diagonal matrix with weights on its diagonal.

    Parameters
    ----------

    y : np.float(y)
        The target, in the model $y = X\beta$

    X : np.float((n, p))
        The data, in the model $y = X\beta$

    weights : np.float
        Coefficients of the L-1 penalty in
        optimization problem, note that different
        coordinates can have different coefficients.

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

            beta_interval = lambda a, b: betainc(0.5, self.df_E*0.5,
                                                 a, b,
                                                 regularized=True)

            self.P_E = np.dot(X_E, self._XEinv)
            self.R_E = np.identity(n) - self.P_E

            _denE = np.sqrt(1 - np.linalg.norm(self.w_E)**2)
            c_E = np.linalg.norm(y - np.dot(self.P_E, y)) / _denE

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
                                                        c_E * self.weights,
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
#         if not hasattr(self, "_intervals"):
#             self._intervals = []
#             C = self.active_constraints
#             XAinv = self._XAinv
#             if XAinv is not None:
#                 for i in range(XAinv.shape[0]):
#                     eta = XAinv[i]
#                     _interval = C.interval(eta, self.y,
#                                            alpha=self.alpha,
#                                            UMAU=self.UMAU)
#                     self._intervals.append((self.active[i], eta, 
#                                             (eta*self.y).sum(), 
#                                             _interval))
        return self._intervals

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

#    @property 
#    def active_intervals(self):
#        if not hasattr(self,"_intervals"):
#            self._intervals = None
#            if self.active.shape[0] > 0:
#                self._intervals = []
#                C = self.active_constraints
#                XEinv = self._XEinv
#                if XEinv is not None:
#                    for i in range(XEinv.shape[0]):
#                        eta = XEinv[i]
#
#                        def p_value(theta):
#                            (intervals, Tobs) = constraints_unknown_sigma( \
#                                C.linear_part,
#                                C.offset / self.sigma_E,
#                                self.y,
#                                eta,
#                                self.R_E, value_under_null=theta)
#                            truncT = truncated_T(np.array([(interval.lower_value,
#                                                            interval.upper_value) for interval in intervals]), self.df_E)
#                            if (truncT.intervals.shape == ((1,2)) and np.all(truncT.intervals == [[-np.inf, np.inf]])):
#                                raise ValueError('should be truncated')
#
#                            return truncT.cdf(Tobs)
#
#                        lower = find_root(p_value, 0.025, -100, 100)
#                        upper = find_root(p_value, 0.975, -100, 100)
#                        self._intervals.append((self.active[i], (lower, upper)))
#        return self._intervals

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
                        _interval = C.interval(eta, self.y)
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
                        _interval = C.interval(eta, self.y)
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


