import numpy as np
from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from pvalues_fixedX import pval
from matplotlib import pyplot as plt
import regreg.api as rr

def test_lasso(s=3, n=200, p=10):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    print 'sigma', sigma
    lam_frac = 1.

    randomization = laplace(loc=0, scale=1.)
    loss = randomized.gaussian_Xfixed(X, y)

    random_Z = randomization.rvs(p)
    epsilon = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))
    random_Z = randomization.rvs(p)

    groups = [0,0,1,1,2,2,3,3,4,4]
    ngroups  = np.unique(groups).shape[0]
    print 'ngroups', ngroups

    groups_mat = np.zeros((p, ngroups), dtype=bool)
    weights = {}
    for g in range(ngroups):
        groups_mat[:, g] = [(groups[j] == g) for j in range(p)]
        x = np.sqrt(np.sum(groups_mat[:,g]))
        weights.update({g: x})

    penalty = rr.group_lasso(groups, weights, lagrange=lam)

    lambdas = lam*np.ones(ngroups)
    for g in range(ngroups):
        lambdas[g] *= weights[g]

    # initial solution
    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0,
                                        -random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_sol = problem.solve(random_term, **solve_args)

    active_groups = np.zeros(ngroups, dtype = bool)
    active_vars = np.zeros(p, dtype = bool)

    for g in range(ngroups):
        if np.sum(initial_sol[groups_mat[:,g]]!=0)>0:
            active_groups[g] = True
            active_vars[groups_mat[:, g]] = True

    print 'active_groups', active_groups
    print 'number of nonzero betas', np.sum(initial_sol!=0)
    print 'active variables', active_vars

    subgradient_initial = np.dot(X.T, y-X.dot(initial_sol)) + random_Z -epsilon*initial_sol

    gamma = np.zeros(ngroups)
    for g in range(ngroups):
        gamma[g] = np.linalg.norm(initial_sol[groups_mat[:, g]])

    data = y.copy()
    ndata = data.shape[0]

    nactive_groups = np.sum(active_groups)
    nactive_vars = np.sum(active_vars)
    inactive_vars = ~active_vars
    ninactive_vars = np.sum(inactive_vars)
    init_vec_state = np.zeros(ndata+nactive_groups+ninactive_vars)
    init_vec_state[:ndata] = data.copy()
    init_vec_state[ndata:(ndata+nactive_groups)] = gamma[active_groups].copy()
    init_vec_state[(ndata+nactive_groups):] = subgradient_initial[inactive_vars].copy()


    def full_projection(vec_state, lambdas = lambdas,
                        active_groups = active_groups, groups_mat = groups_mat,
                        ndata=ndata):

        data = vec_state[:ndata].copy()
        gamma_active = vec_state[ndata:(ndata+nactive_groups)]
        subgradient_inactive = vec_state[(ndata+nactive_groups):]

        projected_gamma_active = gamma_active.copy()
        projected_subgradient_inactive = subgradient_inactive.copy()

        projected_gamma_active = np.clip(projected_gamma_active, 0, np.inf)

        inactive_groups_mat = groups_mat[~active_vars,:]
        inactive_groups = ~active_groups
        inactive_groups_set = np.where(inactive_groups)[0]
        for i, g in enumerate(inactive_groups_set):
            g_set = inactive_groups_mat[:, g] # part of inactive variables in g
            z_g = subgradient_inactive[g_set]
            z_g_norm = np.linalg.norm(z_g)
            if (z_g_norm>lambdas[g]):
                projected_subgradient_inactive[g_set] = z_g*(lambdas[g]/z_g_norm)

        return np.concatenate((data, projected_gamma_active, projected_subgradient_inactive), 0)


    def mat_gamma(X=X, lambdas=lambdas,
                       subgradient_initial = subgradient_initial,
                       active_groups=active_groups, active_vars=active_vars,
                       groups_mat=groups_mat,
                       epsilon=epsilon, ndata=ndata, p=p):

        mat1 = np.dot(X.T, X) + epsilon*np.identity(X.shape[1])

        vec_z = np.zeros(nactive_vars)

        col_ones = np.zeros((nactive_vars, nactive_groups))

        active_groups_set = np.where(active_groups)[0]
        active_groups_mat = groups_mat[active_vars,:]
        subgradient_active = subgradient_initial[active_vars]

        for i, g in enumerate(active_groups_set):
            g_set = active_groups_mat[:, g]  # part of active vars in g
            vec_z[g_set] = subgradient_active[g_set] / lambdas[g]
            vec = np.zeros(nactive_vars)
            vec[g_set] = 1
            col_ones[:, i] = vec.copy()

        return  np.dot(mat1[:, active_vars], np.diag(vec_z)).dot(col_ones)

    _mat_gamma = mat_gamma()


    def gradient_log_jac(gamma_active):

        active_groups_set = np.where(active_groups)[0]
        active_groups_mat = groups_mat[active_vars, :]
        subgradient_active = subgradient_initial[active_vars]

        nactive_groups=np.sum(active_groups)

        D = np.zeros((nactive_vars, nactive_vars))
        XE = X[:,active_vars]
        mat1 = np.dot(XE.T, XE) + epsilon*np.identity(nactive_vars)
        D_seq = np.zeros((nactive_groups, nactive_vars, nactive_vars))
        for i, g in enumerate(active_groups_set):
            g_set = active_groups_mat[:, g]  # part of active vars in g
            z_g = subgradient_active[g_set]
            D_seq[i][:, g_set] = mat1[:, g_set]
            col_block = gamma_active[i] * np.copy(mat1[:, g_set])
            col_block[g_set, :] += lambdas[g]*np.identity(np.sum(g_set)) - (np.outer(z_g, z_g)/lambdas[g])
            D[:, g_set] = np.copy(col_block)

        _gradient_log_jac = np.zeros(nactive_groups)

        D_inv = np.linalg.inv(D)

        for i in range(nactive_groups):
            _gradient_log_jac[i] = + np.sum(np.diag(D_inv.dot(D_seq[i])))
            if gamma_active[i]>0:
                _gradient_log_jac[i] -= (1./gamma_active[i])
            else:
                _gradient_log_jac[i] -= 10

        return _gradient_log_jac


    def full_gradient(vec_state, X=X, subgradient_initial=subgradient_initial,
                      active_groups = active_groups, active_vars=active_vars,
                      groups_mat = groups_mat, lambdas = lambdas,
                      epsilon=epsilon, ndata=ndata, p=p):

        data = vec_state[:ndata]
        gamma_active = vec_state[ndata:(ndata + nactive_groups)]
        subgradient_inactive = vec_state[(ndata + nactive_groups):]

        subgradient_full = np.zeros(p)
        subgradient_full[inactive_vars] = subgradient_inactive
        subgradient_full[active_vars] = subgradient_initial[active_vars]


        w = -np.dot(X.T, y) + subgradient_full + _mat_gamma.dot(gamma_active)
        sign_vec = np.sign(w)

        mat_z = np.zeros((p, ninactive_vars))
        mat_z[inactive_vars,:] = np.identity(ninactive_vars)

        _gradient = np.zeros(ndata + nactive_groups+ninactive_vars)
        _gradient[:ndata] = - (data - np.dot(X, sign_vec))
        _gradient_log_jacobian = gradient_log_jac(gamma_active)
        _gradient[ndata:(ndata + nactive_groups)] = - np.dot(_mat_gamma.T, sign_vec) + _gradient_log_jacobian
        _gradient[(ndata + nactive_groups):] = - np.dot(mat_z.T, sign_vec)

        return _gradient


    null, alt = pval(init_vec_state, full_gradient, full_projection,
                      X, y, nonzero, active_vars)

    return null, alt

if __name__ == "__main__":

    P0, PA = [], []
    for i in range(20):
        print "iteration", i
        p0, pA = test_lasso()
        P0.extend(p0); PA.extend(pA)

    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0,1], color='k', linestyle='-', linewidth=2)
    plt.show()