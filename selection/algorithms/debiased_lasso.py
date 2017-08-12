import numpy as np
from regreg.api import (quadratic_loss,
                        identity_quadratic,
                        l1norm,
                        simple_problem)

from ..constraints.affine import constraints

def _find_row_approx_inverse(Sigma, j, delta, solve_args={'min_its':100, 'tol':1.e-6, 'max_its':500}):
    """

    Find an approximation of j-th row of inverse of Sigma.

    Solves the problem

    .. math::

        \text{min}_{\theta} \frac{1}{2} \theta^TS\theta

    subject to $\|\Sigma \hat{\theta} - e_j\|_{\infty} \leq \delta$ with
    $e_j$ the $j$-th elementary basis vector and `S` as $\Sigma$, 
    and `delta` as $\delta$.

    Described in Table 1, display (4) of https://arxiv.org/pdf/1306.3171.pdf

    """
    p = Sigma.shape[0]
    elem_basis = np.zeros(p, np.float)
    elem_basis[j] = 1.
    loss = quadratic_loss(p, Q=Sigma)
    penalty = l1norm(p, lagrange=delta)
    iq = identity_quadratic(0, 0, elem_basis, 0)
    problem = simple_problem(loss, penalty)
    dual_soln = problem.solve(iq, **solve_args)

    soln = -dual_soln

    # check feasibility -- if it fails miserably
    # presume delta was too small

    feasibility_gap = np.fabs(Sigma.dot(soln) - elem_basis).max()
    if feasibility_gap > (1.01) * delta:
        raise ValueError('does not seem to be a feasible point -- try increasing delta')

    return soln


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
        raise ValueError('debiased lasso should be fit ignoring active constraints as implied covariance between active and inactive score is 0')

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

    _full_linear_part[:_constraints.linear_part.shape[0]][:,:_constraints.linear_part.shape[1]] = _constraints.linear_part
    _full_linear_part[_constraints.linear_part.shape[0]:][:,_constraints.linear_part.shape[1]:] = _inactive_constraints.linear_part

    _full_offset = np.zeros(_full_linear_part.shape[0])
    _full_offset[:_constraints.linear_part.shape[0]] = _constraints.offset
    _full_offset[_constraints.linear_part.shape[0]:] = _inactive_constraints.offset

    _full_cov = np.zeros((_full_linear_part.shape[1],
                          _full_linear_part.shape[1]))
    _full_cov[:_constraints.linear_part.shape[1]][:,:_constraints.linear_part.shape[1]] = _constraints.covariance
    _full_cov[_constraints.linear_part.shape[1]:][:,_constraints.linear_part.shape[1]:] = _inactive_constraints.covariance
    _full_constraints = constraints(_full_linear_part,
                                    _full_offset,
                                    covariance=_full_cov)
                                    
    _full_data = np.hstack([lasso_active, G_I])
    if not _full_constraints(_full_data):
        raise ValueError('constraints not satisfied')

    H = lasso_obj.loglike.hessian(lasso_obj.lasso_solution)
    H_AA = H[lasso_obj.active][:,lasso_obj.active]
    bias_AA = np.linalg.inv(H_AA).dot(subgrad_term)

    intervals = []
    pvalues = []
    for var in variables:
        theta_var = _find_row_approx_inverse(H, var, delta)

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
