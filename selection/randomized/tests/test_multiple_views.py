import numpy as np

import regreg.api as rr

from selection.api import randomization, glm_group_lasso, pairs_bootstrap_glm, multiple_views, discrete_family, projected_langevin, glm_group_lasso_parametric
from selection.randomized.tests import logistic_instance, wait_for_return_value
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

#@wait_for_return_value
def test_multiple_views():
    s, n, p = 3, 100, 10

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=3)

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
    #mv = multiple_views([M_est1])
    mv.solve()

    active_union = M_est1.overall + M_est2.overall
    nactive = np.sum(active_union)
    print "nactive", nactive

    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        inactive_indicators = np.zeros(nactive)
        for i in range(nactive):
            if active_set[i] not in nonzero:
                inactive_indicators[i] = 1

        inactive_indicators_mat = np.zeros((len(inactive_selected),nactive))
        j = 0
        for i in range(nactive):
            if active_set[i] not in nonzero:
                inactive_indicators_mat[j,i] = 1
                j+=1

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        boot_target, target_observed = pairs_bootstrap_glm(loss, active_union)
        inactive_target = lambda indices: boot_target(indices)[inactive_selected]
        inactive_observed = target_observed[inactive_selected]
        # param_cov = _parametric_cov_glm(loss, active_union)

        alpha_mat = set_alpha_matrix(loss, active_union)
        target_alpha = np.dot(inactive_indicators_mat, alpha_mat) # target = target_alpha\times alpha+reference_vec

        target_sampler = mv.setup_target(inactive_target, inactive_observed, n, target_alpha)

        test_stat = lambda x: np.linalg.norm(x)
        test_stat_boot = lambda x: np.linalg.norm(np.dot(target_alpha, x))
        pval = target_sampler.boot_hypothesis_test(test_stat_boot, np.linalg.norm(inactive_observed), alternative='twosided')


        # testing the global null
        all_selected = np.arange(active_set.shape[0])
        target_gn = lambda indices: boot_target(indices)[:nactive]
        target_observed_gn = target_observed[:nactive]

        target_alpha_gn = alpha_mat

        target_sampler_gn = mv.setup_target(target_gn, target_observed_gn, n, target_alpha_gn, reference = beta[active_union])
        test_stat_boot_gn = lambda x: np.linalg.norm(np.dot(target_alpha_gn, x))
        pval_gn = target_sampler_gn.boot_hypothesis_test(test_stat_boot_gn, np.linalg.norm(target_observed_gn), alternative='twosided')

        return pval, pval_gn


def test_multiple_views_individual_coeff():
    s, n, p = 0, 100, 10

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=3)

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
    #mv = multiple_views([M_est1])
    mv.solve()

    active_union = M_est1.overall + M_est2.overall
    nactive = np.sum(active_union)
    print "nactive",nactive
    active_set = np.nonzero(active_union)[0]

    pvalues = []
    true_beta = beta[active_union]
    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)
        boot_target, target_observed = pairs_bootstrap_glm(loss, active_union)
        alpha_mat = set_alpha_matrix(loss, active_union)

        for j in range(nactive):

            individual_target = lambda indices: boot_target(indices)[j]
            individual_observed = target_observed[j]
            # param_cov = _parametric_cov_glm(loss, active_union)

            target_alpha = np.atleast_2d(alpha_mat[j,:]) # target = target_alpha\times alpha+reference_vec

            target_sampler = mv.setup_target(individual_target, individual_observed, n, target_alpha, reference=true_beta[j])

            test_stat_boot = lambda x: np.inner(target_alpha, x)
            pval = target_sampler.boot_hypothesis_test(test_stat_boot, individual_observed, alternative='twosided')
            pvalues.append(pval)


        return pvalues




@wait_for_return_value
def test_parametric_covariance():
    s, n, p = 3, 100, 30

    randomizer = randomization.laplace((p,), scale=1)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=5)

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


import matplotlib.pyplot as plt
from scipy.stats import probplot, uniform
import statsmodels.api as sm


def make_a_plot():

    np.random.seed(22)
    fig = plt.figure()
    fig.suptitle('Pivots for the simple example wild bootstrap')

    pvalues = []
    pvalues_gn = []
    for i in range(100):
        print "iteration", i
        pvals = test_multiple_views()
        if pvals is not None:
            pval, pval_gn = pvals
            pvalues.append(pval)
            pvalues_gn.append(pval_gn)
            print "pvalue", pval
            print np.mean(pvalues), np.std(pvalues), np.mean(np.array(pvalues) < 0.05)

    ecdf = sm.distributions.ECDF(pvalues)
    x = np.linspace(min(pvalues), max(pvalues))
    y = ecdf(x)

    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.title("Testing false discoveries")
    #plt.plot(x, y, '-o', lw=2)
    #plt.plot([0, 1], [0, 1], 'k-', lw=1)

    ecdf_gn = sm.distributions.ECDF(pvalues_gn)
    x_gn = np.linspace(min(pvalues_gn), max(pvalues_gn))
    y_gn = ecdf_gn(x_gn)
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.plot(x, y, '-o', lw=2)
    #plt.plot([0, 1], [0, 1], 'k-', lw=1)

    plt.title("Logistic")
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Selected zeros", marker='o', lw=2, markersize=8)
    ax.plot(x_gn, y_gn, label="Global null", marker ='o', lw=2, markersize=8)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.plot([0, 1], [0, 1], 'k-', lw=1)


    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    plt.savefig("Bootstrap after GLM two views")

    #while True:
    #    plt.pause(0.05)
    plt.show()


def make_a_plot_individual_coeff():

    np.random.seed(2)
    fig = plt.figure()
    fig.suptitle('Pivots for the simple example wild bootstrap')

    pvalues = []
    for i in range(50):
        print "iteration", i
        pvals = test_multiple_views_individual_coeff()
        if pvals is not None:
            pvalues.extend(pvals)
            print "pvalues", pvals
            print np.mean(pvalues), np.std(pvalues), np.mean(np.array(pvalues) < 0.05)

    ecdf = sm.distributions.ECDF(pvalues)
    x = np.linspace(min(pvalues), max(pvalues))
    y = ecdf(x)

    plt.title("Logistic")
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Individual coefficients", marker='o', lw=2, markersize=8)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.plot([0, 1], [0, 1], 'k-', lw=1)


    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    plt.savefig("Bootstrap after GLM two views")
    #while True:
    #    plt.pause(0.05)
    plt.show()



make_a_plot()

#make_a_plot_individual_coeff()