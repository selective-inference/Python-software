import numpy as np

import regreg.api as rr

from selection.randomized.randomization import base
from selection.randomized.glm_boot import glm_group_lasso, pairs_bootstrap_glm
from selection.randomized.multiple_views import multiple_views

from selection.algorithms.randomized import logistic_instance
from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin

def test():
    s, n, p = 5, 200, 20

    randomization = base.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    # first randomization
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomization)
    # second randomization
    M_est2 = glm_group_lasso(loss, epsilon, penalty, randomization)

    mv = multiple_views([M_est1, M_est2])
    mv.solve()

#     bootstrap_score = ()
#     active = np.zeros(p, dtype=bool)

#     for i in range(mv.nviews):
#         bootstrap_score += (randomized.glm_boot.pairs_bootstrap_glm(mv.objectives[i].loss,
#                                                                mv.objectives[i].overall,
#                                                                beta_full=mv.objectives[i]._beta_full,
#                                                                # this is private -- we "shouldn't" observe this
#                                                                inactive=mv.objectives[i].inactive)[0],)
#         active += mv.objectives[i].overall



    active = M_est1.overall + M_est2.overall

    if set(nonzero).issubset(np.nonzero(active)[0]):

        boot_target, target_observed = pairs_bootstrap_glm(loss, active)
        mv.setup_sampler((n, n), boot_target, target_observed)

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

#         # target are all true null coefficients selected

#         #target_cov, cov1, cov2 = randomized.glm_boot.bootstrap_cov((n, n), boot_target, cross_terms=bootstrap_score)
#         covariances = randomized.glm_boot.bootstrap_cov((n, n), boot_target, cross_terms=bootstrap_score)

#         target_cov = covariances[0]
#         cov = []
#         for i in range(mv.nviews):
#             cov.append(covariances[i+1])





#         data_transform = ()
#         for i in range(mv.nviews):
#             data_transform += (mv.objectives[i].condition(cov[i][I], target_cov[I][:,I], target_observed[I]),)


#         target_inv_cov = np.linalg.inv(target_cov[I][:,I])

#         initial_state = np.hstack([target_observed[I].copy(),
#                                    mv.observed_opt_state])

#         ntarget = len(I)
#         target_slice = slice(0, ntarget)
#         opt_slice = slice(ntarget, mv.num_opt_var + ntarget)

#         def target_gradient(state):

#             target = state[target_slice]
#             opt_state = state[opt_slice]
#             target_grad = mv.gradient(target, data_transform, opt_state)

#             full_grad = np.zeros_like(state)
#             full_grad[opt_slice] = -target_grad[1]
#             full_grad[target_slice] -= target_grad[0]
#             full_grad[target_slice] -= target_inv_cov.dot(target)

#             return full_grad

#         def target_projection(state):
#             opt_state = state[opt_slice]
#             state[opt_slice] = mv.projection(opt_state)
#             return state


        target_langevin = projected_langevin(mv.observed_state.copy(),
                                             mv.gradient,
                                             mv.projection,
                                             .5 / (2*p + 1))


        Langevin_steps = 20000
        burning = 10000
        samples = []
        for i in range(Langevin_steps):
            if (i>=burning):
                target_langevin.next()
                samples.append(target_langevin.state[mv.target_slice].copy())

        test_stat = lambda x: np.linalg.norm(x)
        observed = test_stat(target_observed[I])
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.ccdf(0, observed)
        print "pvalue", pval
        return pval


if __name__ == "__main__":

    pvalues = []
    for i in range(50):
        print "iteration", i
        pval = test()
        if pval >-1:
            pvalues.append(pval)

    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    probplot(pvalues, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.pause(0.01)

    while True:
        plt.pause(0.05)
    plt.show()

