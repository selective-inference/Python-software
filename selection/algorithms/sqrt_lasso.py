"""
Module to solve sqrt-LASSO convex program using regreg.
"""

import numpy as np
from scipy.stats import norm as ndist, chi as chidist
from scipy.interpolate import interp1d

# regreg http://github.com/regreg 

import regreg.api as rr
import regreg.affine as ra
from regreg.smooth.glm import gaussian_loglike

from ..constraints.affine import (constraints as affine_constraints, 
                                  sample_from_sphere)
from ..distributions.discrete_multiparameter import multiparameter_family
from ..distributions.discrete_family import discrete_family

class sqlasso_objective(rr.smooth_atom):
    """

    The square-root LASSO objective. Essentially
    smooth, but singular on 
    $\{\beta: y=X\beta\}$.

    This singularity is ignored in solving the problem.
    It might be a problem sometimes?

    """

    _sqrt2 = np.sqrt(2) # often used constant

    def __init__(self, X, Y, 
                 quadratic=None, 
                 initial=None,
                 offset=None):

        X = rr.astransform(X)
        rr.smooth_atom.__init__(self,
                                X.input_shape,
                                coef=1.,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

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

def solve_sqrt_lasso(X, Y, weights=None, initial=None, quadratic=None, solve_args={}):
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

    solve_args : dict
        Arguments passed to regreg solver.

    quadratic : `regreg.identity_quadratic`
        A quadratic term added to objective function.
    """
    n, p = X.shape
    if n > p:
        return solve_sqrt_lasso_skinny(X, Y, weights=weights, initial=initial, quadratic=quadratic, solve_args=solve_args)
    else:
        return solve_sqrt_lasso_fat(X, Y, weights=weights, initial=initial, quadratic=quadratic, solve_args=solve_args)

def solve_sqrt_lasso_fat(X, Y, weights=None, initial=None, quadratic=None, solve_args={}):
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

    solve_args : dict
        Arguments passed to regreg solver.

    quadratic : `regreg.identity_quadratic`
        A quadratic term added to objective function.

    """
    n, p = X.shape
    if weights is None:
        lam = choose_lambda(X)
        weights = lam * np.ones((p,))

    loss = sqlasso_objective(X, Y)
    penalty = rr.weighted_l1norm(weights, lagrange=1.)
    problem = rr.simple_problem(loss, penalty)
    if initial is not None:
        problem.coefs[:] = initial
    soln = problem.solve(quadratic, **solve_args)
    return soln, loss

class sqlasso_objective_skinny(rr.smooth_atom):
    """

    The square-root LASSO objective on larger parameter space:

    .. math::

         (\beta, \sigma) \mapsto \frac{\|y-X\beta\|_2^2}{\sigma} + \sigma

    """

    def __init__(self, X, Y):

        self.X = X
        n, p = X.shape 
        self.Y = Y
        self._constant_term = (Y**2).sum()
        if n > p:
            self._quadratic_term = X.T.dot(X)
            self._linear_term = -2 * X.T.dot(Y)
        self._sqerror = rr.squared_error(X, Y)

    def smooth_objective(self, x, mode='both', check_feasibility=False):

        n, p = self.X.shape

        beta, sigma = x[:p], x[p]

        if n > p:
            if mode in ['grad', 'both']:
                g = np.zeros(p+1)
                g0 = self._quadratic_term.dot(beta) 
                f1 = self._constant_term + (self._linear_term * beta).sum() + (g0 * beta).sum()
                g1 = 2 * g0 + self._linear_term
            else:
                g1 = self._quadratic_term.dot(beta)
                f1 = self._constant_term + (self._linear_term * beta).sum() + (g1 * beta).sum()
        else:
            if mode in ['grad', 'both']:
                g = np.zeros(p+1)
                f1, g1 = self._sqerror.smooth_objective(beta, 'both')
                f1 *= 2; g1 *= 2
            else:
                f1 = self._sqerror.smooth_objective(beta, 'func')
                f1 *= 2

        f = f1 / sigma + sigma

        if mode == 'both':
            g[:p] = g1 / sigma
            g[p] = -f1 / sigma**2 + 1.
            return f, g
        elif mode == 'grad':
            g[:p] = g1 / sigma
            g[p] = -f1 / sigma**2 + 1.
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")

def solve_sqrt_lasso_skinny(X, Y, weights=None, initial=None, quadratic=None, solve_args={}):
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

    solve_args : dict
        Arguments passed to regreg solver.

    quadratic : `regreg.identity_quadratic`
        A quadratic term added to objective function.

    """
    X = rr.astransform(X)
    n, p = X.output_shape[0], X.input_shape[0]
    if weights is None:
        lam = choose_lambda(X)
        weights = lam * np.ones((p,))
    weight_dict = dict(zip(np.arange(p),
                           2 * weights))
    penalty = rr.mixed_lasso(list(np.arange(p)) + [rr.NONNEGATIVE], lagrange=1.,
                             weights=weight_dict)

    loss = sqlasso_objective_skinny(X, Y)
    problem = rr.simple_problem(loss, penalty)
    problem.coefs[-1] = np.linalg.norm(Y)
    if initial is not None:
        problem.coefs[:-1] = initial

    if quadratic is not None:
        collapsed = quadratic.collapsed()
        new_linear_term = np.zeros(p+1)
        new_linear_term[:p] = collapsed.linear_term
        new_quadratic = rr.identity_quadratic(collapsed.coef, 
                                              0., 
                                              new_linear_term, 
                                              collapsed.constant_term)
    else:
        new_quadratic = None

    soln = problem.solve(new_quadratic, **solve_args)
    _loss = sqlasso_objective(X, Y)
    return soln[:-1], _loss

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

    if untruncated_df < 50:
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
    # but this factor seems to correct it.
    # it is such that if there were no selection it would be 
    # the usual unbiased estimate

    try:
        sigma_hat = np.min(V[interpolant(V) >= observed**2 * total_df + observed**2 * linear_term])
    except ValueError:
        # no solution, just return observed
        sigma_hat = observed
        
    return sigma_hat

def goodness_of_fit(lasso_obj, statistic, 
                    force=False,
                    alternative='twosided', 
                    ndraw=5000,
                    burnin=2000,
                    sample=None,
                    ):

    """

    Compute a goodness of fit test based on a given
    statistic applied to 

    .. math::

         U_{-E}(y) = (I-P_E)y / \|(I-P_E)y\|_2

    which is ancillary for the selected model.

    Parameters
    ----------

    lasso_obj : `lasso`
        Instance of selection.algorithms.lasso.lasso instantiated
        with a gaussian loglikelihood (instance of `regreg.smooth.glm.gaussian_loglike`

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

    sample : multiparameter_family (optional)
        If not None, this is used as sample instead of generating a new sample.

    Returns
    -------

    pvalue : np.float
         Two-tailed p-value.

    """

    L = lasso_obj # shorthand
    if not isinstance(L.loglike.saturated_loss, gaussian_loglike):
        raise ValueError('goodness of fit test assumes response is Gaussian')

    X, Y = L.loglike.data
    n, p = X.shape

    if len(lasso_obj.active) > 0:
        X_E = X[:,L.active]
        C_Ei = np.linalg.pinv(X_E.T.dot(X_E))
        R_E = lambda z: z - X_E.dot(C_Ei.dot(X_E.T.dot(z)))

        X_minus_E = X[:,L.inactive]
        RX_minus_E = R_E(X_minus_E)
        inactive_bound = L.feature_weights[L.inactive]
        active_subgrad = L.feature_weights[L.active] * L.active_signs
        irrep_term = X_minus_E.T.dot(X_E.dot(C_Ei.dot(active_subgrad)))

        inactive_constraints = affine_constraints(
                                 np.vstack([RX_minus_E.T,
                                            -RX_minus_E.T]),
                                 np.hstack([inactive_bound - irrep_term,
                                            inactive_bound + irrep_term]),
                                 covariance = np.identity(n)) # because we condition on norm, covariance doesn't matter

    if sample is None:
        if len(lasso_obj.active) > 0:
            conditional_con = inactive_constraints.conditional(X_E.T, X_E.T.dot(Y))

            Z, W = sample_from_sphere(conditional_con, 
                                      Y,
                                      ndraw=ndraw,
                                      burnin=burnin)  
            U_notE_sample = R_E(Z.T).T
            U_notE_sample /= np.sqrt((U_notE_sample**2).sum(1))[:,None]
            _goodness_of_fit_sample = multiparameter_family(U_notE_sample, W)
            _goodness_of_fit_observed = R_E(Y) 
            _goodness_of_fit_observed /= np.linalg.norm(_goodness_of_fit_observed)

        else:
            U_sample = np.random.standard_normal((ndraw, n))
            U_sample /= np.sqrt((U_sample**2).sum(1))[:, None]
            _goodness_of_fit_sample = multiparameter_family(U_sample, np.ones(U_sample.shape[0]))
            _goodness_of_fit_observed = Y / np.linalg.norm(Y)
    else:
        _goodness_of_fit_sample = sample

    null_sample = _goodness_of_fit_sample.sufficient_stat
    importance_weights = _goodness_of_fit_sample.weights
    null_statistic = np.array([statistic(u) for u in null_sample])
    observed = statistic(_goodness_of_fit_observed)
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

