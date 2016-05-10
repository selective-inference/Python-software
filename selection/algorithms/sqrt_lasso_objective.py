"""
Module to solve sqrt-LASSO convex program using regreg.
"""

import numpy as np
from scipy.stats import norm as ndist, chi as chidist
from scipy.interpolate import interp1d

# regreg http://github.com/regreg 

import regreg.api as rr

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
    if n < p:
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
    soln = problem.solve(quadratic, **solve_args)
    return soln, loss

class sqlasso_objective_skinny(rr.smooth_atom):
    """

    The square-root LASSO objective on larger parameter space:

    .. math::

         (\beta, \sigma) \mapsto \frac{\|y-X\beta\|_2^2}{\sigma} + \sigma

    """

    def __init__(self, X, Y):

        self.X = rr.astransform(X)
        n, p = self.X.output_shape[0], self.X.input_shape[0]

        self.Y = Y
        if n > p:
            self._quadratic_term = np.dot(X.T, X)
            self._linear_term = -2 * np.dot(X.T, Y)
            self._constant_term = (Y**2).sum()
        self._sqerror = rr.squared_error(X, Y)

    def smooth_objective(self, x, mode='both', check_feasibility=False):

        n, p = self.X.output_shape[0], self.X.input_shape[0]

        beta, sigma = x[:p], x[p]

        if n > p:
            if mode in ['grad', 'both']:
                g = np.zeros(p+1)
                g0 = np.dot(self._quadratic_term, beta) 
                f1 = self._constant_term + (self._linear_term * beta).sum() + (g0 * beta).sum()
                g1 = 2 * g0 + self._linear_term
            else:
                g1 = np.dot(self._quadratic_term, beta)
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
    n, p = X.shape
    if weights is None:
        lam = choose_lambda(X)
        weights = lam * np.ones((p,))
    weight_dict = dict(zip(np.arange(p),
                           2 * weights))
    penalty = rr.mixed_lasso(range(p) + [rr.NONNEGATIVE], lagrange=1.,
                             weights=weight_dict)

    loss = sqlasso_objective_skinny(X, Y)
    problem = rr.simple_problem(loss, penalty)
    problem.coefs[-1] = np.linalg.norm(Y)
    if initial is not None:
        problem.coefs[:-1] = initial
    soln = problem.solve(quadratic, **solve_args)
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
