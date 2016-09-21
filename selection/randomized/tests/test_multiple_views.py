import numpy as np

import regreg.api as rr

from selection.api import randomization, glm_group_lasso, pairs_bootstrap_glm, multiple_views, discrete_family, projected_langevin, glm_group_lasso_parametric
from selection.randomized.tests import logistic_instance, wait_for_return_value
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

@wait_for_return_value
def test_multiple_views():
    s, n, p = 3, 200, 20

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=0)

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
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    # second randomization
    M_est2 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est1, M_est2])
    mv.solve()

    active_union = M_est1.overall + M_est2.overall
    nactive = np.sum(active_union)
    active_individual = [M_est1.overall, M_est2.overall]

    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        inactive_indicators = np.zeros(nactive)
        for i in range(nactive):
            if active_set[i] not in nonzero:
                inactive_indicators[i] = 1

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        boot_target, target_observed = pairs_bootstrap_glm(loss, active_union)
        inactive_target = lambda indices: boot_target(indices)[inactive_selected]
        inactive_observed = target_observed[inactive_selected]
        # param_cov = _parametric_cov_glm(loss, active_union)

        alpha_mat = set_alpha_matrix(loss, active_union)
        target_alpha = np.dot(np.diag(inactive_indicators), alpha_mat) # target = target_alpha\times alpha+reference_vec

        #print target_alpha
        target_sampler = mv.setup_target(inactive_target, inactive_observed, n, target_alpha)

        test_stat = lambda x: np.linalg.norm(x)
        test_stat_boot = lambda x: np.linalg.norm(np.dot(target_alpha, x))
        pval = target_sampler.boot_hypothesis_test(test_stat_boot, np.linalg.norm(inactive_observed), alternative='greater')

        return pval


@wait_for_return_value
def test_parametric_covariance():
    s, n, p = 5, 100, 30

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=8)

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
    M_est1 = glm_group_lasso_parametric(loss, epsilon, penalty, randomizer)
    # second randomization
    M_est2 = glm_group_lasso_parametric(loss, epsilon, penalty, randomizer)

    mv = multiple_views([M_est1, M_est2])
    mv.solve()

    target = M_est1.overall.copy()
    if target[-1] or M_est2.overall[-1]:
        return None
    if target[-2] or M_est2.overall[-2]:
        return None
    # we should check they are different sizes
    target[-2:] = 1

    if set(nonzero).issubset(np.nonzero(target)[0]):

        form_covariances = glm_parametric_covariance(loss)
        mv.setup_sampler(form_covariances)

        target_observed = restricted_Mest(loss, target)
        linear_func = np.zeros((2,target_observed.shape[0]))
        linear_func[0,-1] = 1. # we know this one is null
        linear_func[1,-2] = 1. # also null

        target_observed = linear_func.dot(target_observed)
        target_sampler = mv.setup_target((target, linear_func), target_observed)

        test_stat = lambda x: np.linalg.norm(x)
        pval = target_sampler.hypothesis_test(test_stat, target_observed, alternative='greater')

        return pval

def make_a_plot():

    pvalues = []
    for i in range(100):
        print "iteration", i
        pval = test_multiple_views()
        if pval >-1:
            pvalues.append(pval)
            print np.mean(pvalues), np.std(pvalues), np.mean(np.array(pvalues) < 0.05)

    import matplotlib.pyplot as plt
    from scipy.stats import probplot, uniform

    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    probplot(pvalues, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.pause(0.01)

    while True:
        plt.pause(0.05)
    plt.show()


make_a_plot()