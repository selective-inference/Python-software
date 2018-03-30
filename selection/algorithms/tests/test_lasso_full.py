import numpy as np

from ..lasso import lasso_full

def solve_problem(Qbeta_bar, Q, lagrange, initial=None):
    p = Qbeta_bar.shape[0]
    loss = rr.quadratic_loss((p,), Q=Q, quadratic=rr.identity_quadratic(0, 
                                                                        0, 
                                                                        Qbeta_bar, 
                                                                        0))
    lagrange = np.asarray(lagrange)
    if lagrange.shape in [(), (1,)]:
        lagrange = np.ones(p) * lagrange
    pen = rr.weighted_l1norm(lagrange, lagrange=1.)
    problem = rr.simple_problem(loss, pen)
    if initial is not None:
        problem.coefs[:] = initial
    soln = problem.solve(tol=1.e12, min_its=10)
    return soln

def truncation_interval(Qbeta_bar, Q, Qi_jj, j, beta_barj, lagrange):
    if lagrange[j] != 0:
        lagrange_cp = lagrange.copy()
    lagrange_cp[j] = np.inf
    restricted_soln = solve_problem(Qbeta_bar, Q, lagrange_cp)

    p = Qbeta_bar.shape[0]
    I = np.identity(p)
    nuisance = Qbeta_bar - I[:,j] / Qi_jj * beta_barj
    
    center = nuisance[j] - Q[j].dot(restricted_soln)
    upper = (lagrange[j] - center) * Qi_jj
    lower = (lagrange[j] - center) * Qi_jj

    return lower, upper
