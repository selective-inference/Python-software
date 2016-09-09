import numpy as np

import regreg.api as rr

from selection.randomized.randomization import base
from selection.randomized.glm_boot import glm_group_lasso, pairs_bootstrap_glm
from selection.randomized.multiple_views import multiple_views

from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin

from . import logistic_instance, wait_for_return_value

@wait_for_return_value
def test_multiple_views():
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

    active = M_est1.overall + M_est2.overall

    if set(nonzero).issubset(np.nonzero(active)[0]):

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        boot_target, target_observed = pairs_bootstrap_glm(loss, active)
        inactive_target = lambda indices: boot_target(indices)[inactive_selected]
        inactive_observed = target_observed[inactive_selected]
        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        mv.setup_sampler(sampler) 
        target_sampler = mv.setup_target(inactive_target, inactive_observed)
        test_stat = lambda x: np.linalg.norm(x)

        pval = target_sampler.hypothesis_test(test_stat, inactive_observed, alternative='greater')
        return pval


if __name__ == "__main__":

    pvalues = []
    for i in range(50):
        print "iteration", i
        pval = test()
        if pval >-1:
            pvalues.append(pval)
            print np.mean(pvalues), np.std(pvalues), np.mean(np.array(pvalues) < 0.05)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    probplot(pvalues, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.pause(0.01)

    while True:
        plt.pause(0.05)
    plt.show()

