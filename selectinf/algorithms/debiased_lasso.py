from warnings import warn

import numpy as np
from scipy.stats import norm as ndist

from ..constraints.affine import constraints
from .debiased_lasso_utils import solve_wide_

def debiasing_matrix(X,
                     rows,
                     bound=None,
                     linesearch=True,  # do a linesearch?
                     scaling_factor=1.5,  # multiplicative factor for linesearch
                     max_active=None,  # how big can active set get?
                     max_try=10,  # how many steps in linesearch?
                     warn_kkt=False,  # warn if KKT does not seem to be satisfied?
                     max_iter=50,  # how many iterations for each optimization problem
                     kkt_stop=True,  # stop based on KKT conditions?
                     parameter_stop=True,  # stop based on relative convergence of parameter?
                     objective_stop=True,  # stop based on relative decrease in objective?
                     kkt_tol=1.e-4,  # tolerance for the KKT conditions
                     parameter_tol=1.e-4,  # tolerance for relative convergence of parameter
                     objective_tol=1.e-4  # tolerance for relative decrease in objective
                     ):
    """
    Find a row of debiasing matrix using line search of
    Javanmard and Montanari.
    """

    n, p = X.shape

    if bound is None:
        orig_bound = (1. / np.sqrt(n)) * ndist.ppf(1. - (0.1 / (p ** 2)))
    else:
        orig_bound = bound

    if max_active is None:
        max_active = max(50, 0.3 * n)

    rows = np.atleast_1d(rows)
    M = np.zeros((len(rows), p))

    nndef_diag = (X ** 2).sum(0) / n

    for idx, row in enumerate(rows):

        bound = orig_bound
        soln = np.zeros(p)
        soln_old = np.zeros(p)
        ever_active = np.zeros(p, np.intp)
        ever_active[0] = row + 1  # C code is 1-based
        nactive = np.array([1], np.intp)

        linear_func = np.zeros(p)
        linear_func[row] = -1
        gradient = linear_func.copy()

        counter_idx = 1
        incr = 0;

        last_output = None

        Xsoln = np.zeros(n)  # X\hat{\beta}

        ridge_term = 0

        need_update = np.zeros(p, np.intp)

        while (counter_idx < max_try):
            bound_vec = np.ones(p) * bound

            result = solve_wide_(X,
                                 Xsoln,
                                 linear_func,
                                 nndef_diag,
                                 gradient,
                                 need_update,
                                 ever_active,
                                 nactive,
                                 bound_vec,
                                 ridge_term,
                                 soln,
                                 soln_old,
                                 max_iter,
                                 kkt_tol,
                                 objective_tol,
                                 parameter_tol,
                                 max_active,
                                 kkt_stop,
                                 objective_stop,
                                 parameter_stop)

            niter = result['iter']

            # Logic for whether we should continue the line search

            if not linesearch: break

            if counter_idx == 1:
                if niter == (max_iter + 1):
                    incr = 1  # was the original problem feasible? 1 if not
                else:
                    incr = 0  # original problem was feasible

            if incr == 1:  # trying to find a feasible point
                if niter < (max_iter + 1) and counter_idx > 1:
                    break
                bound = bound * scaling_factor;
            elif niter == (max_iter + 1) and counter_idx > 1:
                result = last_output  # problem seems infeasible because we didn't solve it
                break  # so we revert to previously found solution

            bound = bound / scaling_factor

            counter_idx += 1
            last_output = {'soln': result['soln'],
                           'kkt_check': result['kkt_check']}

            # If the active set has grown to a certain size
            # then we stop, presuming problem has become
            # infeasible.

            # We revert to the previous solution

            if result['max_active_check']:
                result = last_output
                break

            # Check feasibility

            if warn_kkt and not result['kkt_check']:
                warn("Solution for row of M does not seem to be feasible")
                
        M[idx] = result['soln'] * 1.

    return np.squeeze(M)


def _find_row_approx_inverse_X(X,
                               j,
                               delta,
                               maxiter=50,
                               kkt_tol=1.e-4,
                               objective_tol=1.e-4,
                               parameter_tol=1.e-4,
                               kkt_stop=True,
                               objective_stop=True,
                               parameter_stop=True,
                               max_active=None,
                               ):
    n, p = X.shape
    theta = np.zeros(p)
    theta_old = np.zeros(p)
    X_theta = np.zeros(n)
    linear_func = np.zeros(p)
    linear_func[j] = -1
    gradient = linear_func.copy()
    ever_active = np.zeros(p, np.intp)
    ever_active[0] = j + 1  # C code has ever_active as 1-based
    nactive = np.array([1], np.intp)
    bound = np.ones(p) * delta

    ridge_term = 0

    nndef_diag = (X ** 2).sum(0) / n
    need_update = np.zeros(p, np.intp)

    if max_active is None:
        max_active = max(50, 0.3 * n)

    solve_wide_(X,
                X_theta,
                linear_func,
                nndef_diag,
                gradient,
                need_update,
                ever_active,
                nactive,
                bound,
                ridge_term,
                theta,
                theta_old,
                maxiter,
                kkt_tol,
                objective_tol,
                parameter_tol,
                max_active,
                kkt_stop,
                objective_stop,
                parameter_stop)

    return theta

def pseudoinverse_debiasing_matrix(X,
                                   rows,
                                   tol=1.e-9 # tolerance for rank computaion
                                   ):
    """
    Find a row of debiasing matrix using algorithm of
    Boot and Niedderling from https://arxiv.org/pdf/1703.03282.pdf
    """

    n, p = X.shape
    nactive = len(rows)

    if n < p:

        U, D, V = np.linalg.svd(X, full_matrices=0)
        rank = np.sum(D > max(D) * tol)

        inv_D = 1. / D
        inv_D[rank:] = 0.
        inv_D2 = inv_D**2
        inv = (U * inv_D2[None, :]).dot(U.T)
        scaling = np.zeros(nactive)

        pseudo_XTX = (V.T[rows] * inv_D2[None, :]).dot(V)

        for i in range(nactive):
            var = rows[i]
            scaling[i] = 1. / (X[:,var] * inv.dot(X[:,var]).T).sum()

    else:
        pseudo_XTX = np.linalg.inv(X.T.dot(X))[rows]
        scaling = np.ones(nactive)

    M_active = scaling[:, None] * pseudo_XTX

    return M_active

def debiased_lasso_inference(lasso_obj, variables, delta):
    """
    Debiased estimate is
    .. math::
        \hat{\beta}^d = \hat{\beta} - \hat{\theta} \nabla \ell(\hat{\beta})
    where $\ell$ is the Gaussian loss and $\hat{\theta}$ is an approximation of the
    inverse Hessian at $\hat{\beta}$.
    The term on the right is expressible in terms of the inactive gradient
    as well as the fixed active subgradient. The left hand term is expressible in
    terms of $\bar{\beta}$ the "relaxed" solution and the fixed active subgradient.
    We need a covariance for $(\bar{\beta}_M, G_{-M})$.
    Parameters
    ----------
    lasso_obj : `selection.algorithms.lasso.lasso`
        A lasso object after calling fit() method.
    variables : seq
        Which variables should we produce p-values / intervals for?
    delta : float
        Feasibility parameter for estimating row of inverse of Sigma.
    """

    if not lasso_obj.ignore_inactive_constraints:
        raise ValueError(
            'debiased lasso should be fit ignoring inactive constraints as implied covariance between active and inactive score is 0')

    # should we check that loglike is gaussian

    lasso_soln = lasso_obj.lasso_solution
    lasso_active = lasso_soln[lasso_obj.active]
    active_list = list(lasso_obj.active)

    G = lasso_obj.loglike.smooth_objective(lasso_soln, 'grad')
    G_I = G[lasso_obj.inactive]

    # this is the fixed part of subgradient
    subgrad_term = -G[lasso_obj.active]

    # we make new constraints for the Gaussian vector \hat{\beta}_M --
    # same covariance as those for \bar{\beta}_M, but the constraints are just on signs,
    # not signs after translation

    if lasso_obj.active_penalized.sum():
        _constraints = constraints(-np.diag(lasso_obj.active_signs)[lasso_obj.active_penalized],
                                   np.zeros(lasso_obj.active_penalized.sum()),
                                   covariance=lasso_obj._constraints.covariance)

    _inactive_constraints = lasso_obj._inactive_constraints

    # now make a product of the two constraints
    # assuming independence -- which is true under
    # selected model

    _full_linear_part = np.zeros(((_constraints.linear_part.shape[0] +
                                   _inactive_constraints.linear_part.shape[0]),
                                  (_constraints.linear_part.shape[1] +
                                   _inactive_constraints.linear_part.shape[1])))

    _full_linear_part[:_constraints.linear_part.shape[0]][:,
    :_constraints.linear_part.shape[1]] = _constraints.linear_part
    _full_linear_part[_constraints.linear_part.shape[0]:][:,
    _constraints.linear_part.shape[1]:] = _inactive_constraints.linear_part

    _full_offset = np.zeros(_full_linear_part.shape[0])
    _full_offset[:_constraints.linear_part.shape[0]] = _constraints.offset
    _full_offset[_constraints.linear_part.shape[0]:] = _inactive_constraints.offset

    _full_cov = np.zeros((_full_linear_part.shape[1],
                          _full_linear_part.shape[1]))
    _full_cov[:_constraints.linear_part.shape[1]][:, :_constraints.linear_part.shape[1]] = _constraints.covariance
    _full_cov[_constraints.linear_part.shape[1]:][:,
    _constraints.linear_part.shape[1]:] = _inactive_constraints.covariance
    _full_constraints = constraints(_full_linear_part,
                                    _full_offset,
                                    covariance=_full_cov)

    _full_data = np.hstack([lasso_active, G_I])
    if not _full_constraints(_full_data):
        raise ValueError('constraints not satisfied')

    H = lasso_obj.loglike.hessian(lasso_obj.lasso_solution)
    H_AA = H[lasso_obj.active][:, lasso_obj.active]
    bias_AA = np.linalg.inv(H_AA).dot(subgrad_term)

    intervals = []
    pvalues = []

    approx_inverse = debiasing_matrix(H, variables, delta)

    for Midx, var in enumerate(variables):

        theta_var = approx_inverse[Midx]

        # express target in pair (\hat{\beta}_A, G_I)
        eta = np.zeros_like(theta_var)

        # XXX should be better way to do this
        if var in active_list:
            idx = active_list.index(var)
            eta[idx] = 1.

        # inactive coordinates
        eta[lasso_active.shape[0]:] = theta_var[lasso_obj.inactive]
        theta_active = theta_var[active_list]

        # offset term

        offset = -bias_AA[idx] + theta_active.dot(subgrad_term)

        intervals.append(_full_constraints.interval(eta,
                                                    _full_data) + offset)
        pvalues.append(_full_constraints.pivot(eta,
                                               _full_data,
                                               null_value=-offset,
                                               alternative='twosided'))

    return [(j, p) + tuple(i) for j, p, i in zip(active_list, pvalues, intervals)]
