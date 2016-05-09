"""
Module to solve sqrt-LASSO convex program using regreg.
"""

import numpy as np

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
