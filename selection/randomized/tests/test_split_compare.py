from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.api import randomization, split_glm_group_lasso, pairs_bootstrap_glm, multiple_views, discrete_family, projected_langevin, glm_group_lasso_parametric
from selection.tests.instance import logistic_instance
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

from selection.randomized.multiple_views import naive_confidence_intervals

#@set_seed_iftrue(SET_SEED)
#@set_sampling_params_iftrue(SMALL_SAMPLES)
#@wait_for_return_value()
def test_intervals(ndraw=10000, burnin=2000, nsim=None, solve_args={'min_its':50, 'tol':1.e-10}): # nsim needed for decorator
    s, n, p = 0, 200, 10

    randomizer = randomization.laplace((p,), scale=1.)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=5)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    m = int(0.8 * n)
    # first randomization
    M_est1 = split_glm_group_lasso(loss, epsilon, m, penalty)
    # second randomization
    # M_est2 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    # mv = multiple_views([M_est1, M_est2])
    mv = multiple_views([M_est1])
    mv.solve()

    active_union = M_est1.overall #+ M_est2.overall
    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    leftout_loss = M_est1.leftout_loss

    if set(nonzero).issubset(np.nonzero(active_union)[0]):

        active_set = np.nonzero(active_union)[0]

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        boot_target, target_observed = pairs_bootstrap_glm(loss, active_union)

        # testing the global null
        # constructing the intervals based on the samples of \bar{\beta}_E at the unpenalized MLE as a reference
        all_selected = np.arange(active_set.shape[0])
        target_gn = lambda indices: boot_target(indices)[:nactive]
        target_observed_gn = target_observed[:nactive]

        unpenalized_mle = restricted_Mest(loss, M_est1.overall, solve_args=solve_args)

        alpha_mat = set_alpha_matrix(loss, active_union)
        target_alpha_gn = alpha_mat

        ## bootstrap
        target_sampler_gn_boot = mv.setup_bootstrapped_target(target_gn,
                                                              target_observed_gn,
                                                              n, target_alpha_gn,
                                                              reference = unpenalized_mle)

        target_sample_boot = target_sampler_gn_boot.sample(ndraw=ndraw,
                                                           burnin=burnin)

        LU_boot = target_sampler_gn_boot.confidence_intervals(unpenalized_mle,
                                                         sample=target_sample_boot)

        pvalues_truth_boot = target_sampler_gn_boot.coefficient_pvalues(unpenalized_mle,
                                                              parameter=beta[active_union],
                                                              sample=target_sample_boot)

        ## CLT plugin
        target_sampler_gn = mv.setup_target(target_gn,
                                            target_observed_gn,
                                            reference = unpenalized_mle)

        target_sample = target_sampler_gn.sample(ndraw=ndraw,
                                                 burnin=burnin)


        LU = target_sampler_gn.confidence_intervals(unpenalized_mle,
                                                    sample=target_sample)

        LU_naive = naive_confidence_intervals(target_sampler_gn, unpenalized_mle)

        X2, y2  = leftout_loss.data
        import statsmodels.discrete.discrete_model as sm
        logit = sm.Logit(y2, X2)
        result = logit.fit()
        LU_split = result.conf_int(alpha=0.1)

        boot_leftout_target, leftout_target_observed = pairs_bootstrap_glm(leftout_loss, active_union)
        leftout_size = n-m
        print(leftout_size)
        sampler = lambda: np.random.choice(leftout_size, size=(leftout_size,), replace=True)
        from selection.randomized.glm import bootstrap_cov
        leftout_target_cov = bootstrap_cov(sampler, boot_leftout_target)

        def split_confidence_intervals(observed, cov, alpha=0.1):
            from scipy.stats import norm as ndist
            quantile = - ndist.ppf(alpha / float(2))
            LU = np.zeros((2, observed.shape[0]))
            for j in range(observed.shape[0]):
                sigma = np.sqrt(cov[j, j])
                LU[0, j] = observed[j] - sigma * quantile
                LU[1, j] = observed[j] + sigma * quantile
            return LU.T

        #LU_split = split_confidence_intervals(leftout_target_observed, leftout_target_cov)

        #pvalues_mle = target_sampler_gn.coefficient_pvalues(unpenalized_mle,
        #                                                    parameter=target_sampler_gn.reference,
        #                                                    sample=target_sample)

        pvalues_truth = target_sampler_gn.coefficient_pvalues(unpenalized_mle,
                                                              parameter=beta[active_union],
                                                              sample=target_sample)

        true_vec = beta[active_union]

        def coverage(LU):
            L, U = LU[:,0], LU[:,1]
            print(L,U)
            ncovered = 0
            ci_length = 0

            for j in range(nactive):
                if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                    ncovered += 1
                    ci_length += U[j]-L[j]
            return ncovered, ci_length

        ncovered, ci_length = coverage(LU)
        ncovered_boot, ci_length_boot = coverage(LU_boot)
        ncovered_split, ci_length_split = coverage(LU_split)

        return pvalues_truth, pvalues_truth_boot, ncovered, ci_length, ncovered_boot, ci_length_boot, \
               ncovered_split, ci_length_split, nactive




def make_a_plot():
    import matplotlib.pyplot as plt
    from scipy.stats import probplot, uniform
    import statsmodels.api as sm

    np.random.seed(2)

    _pvalues_truth, _pvalues_truth_boot = [], []
    _nparam = 0
    _ncovered, _ncovered_boot, _ncovered_split = 0, 0, 0
    _ci_length, _ci_length_boot, _ci_length_split = 0, 0, 0

    for i in range(30):
        print("iteration", i)
        test = test_intervals() # first value is a count
        if test is not None:
            pvalues_truth, pvalues_truth_boot, ncovered, ci_length, ncovered_boot, ci_length_boot, \
            ncovered_split, ci_length_split, nactive = test
            _pvalues_truth.extend(pvalues_truth)
            _pvalues_truth_boot.extend(pvalues_truth_boot)
            _nparam += nactive
            _ncovered += ncovered
            _ncovered_boot += ncovered_boot
            _ncovered_split += ncovered_split
            _ci_length += ci_length
            _ci_length_boot += ci_length_boot
            _ci_length_split += ci_length_split
            print("plugin CLT pvalues", np.mean(_pvalues_truth), np.std(_pvalues_truth), np.mean(np.array(_pvalues_truth) < 0.05))
            print("boot pvalues", np.mean(_pvalues_truth_boot), np.std(_pvalues_truth_boot), np.mean(np.array(_pvalues_truth_boot) < 0.05))

        if _nparam > 0:
            print("coverage", _ncovered/float(_nparam))
            print("boot coverage", _ncovered_boot/float(_nparam))
            print("split coverage", _ncovered_split/float(_nparam))

    print("number of parameters", _nparam,"coverage", _ncovered/float(_nparam))
    print("ci length plugin CLT", _ci_length/float(_nparam))
    print("ci length boot", _ci_length_boot / float(_nparam))
    print("ci length split", _ci_length_split / float(_nparam))


    fig = plt.figure()
    fig.suptitle("Pivots at the truth by tilting the samples")
    plot_pvalues_truth = fig.add_subplot(121)
    plot_pvalues_truth_boot = fig.add_subplot(122)

    ecdf_mle = sm.distributions.ECDF(_pvalues_truth)
    x = np.linspace(min(_pvalues_truth), max(_pvalues_truth))
    y = ecdf_mle(x)
    plot_pvalues_truth.plot(x, y, '-o', lw=2)
    plot_pvalues_truth.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues_truth.set_title("Pivots at the truth based on tilting the Gaussian plugin samples")
    plot_pvalues_truth.set_xlim([0, 1])
    plot_pvalues_truth.set_ylim([0, 1])

    ecdf_truth = sm.distributions.ECDF(_pvalues_truth_boot)
    x = np.linspace(min(_pvalues_truth_boot), max(_pvalues_truth_boot))
    y = ecdf_truth(x)
    plot_pvalues_truth_boot.plot(x, y, '-o', lw=2)
    plot_pvalues_truth_boot.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues_truth_boot.set_title("Pivots at the truth by tilting the bootstrapped samples")
    plot_pvalues_truth_boot.set_xlim([0, 1])
    plot_pvalues_truth_boot.set_ylim([0, 1])

    #while True:
    #    plt.pause(0.05)
    plt.show()


def make_a_plot_individual_coeff():

    np.random.seed(3)
    fig = plt.figure()
    fig.suptitle('Pivots for glm wild bootstrap')

    pvalues = []
    for i in range(100):
        print("iteration", i)
        pvals = test_multiple_views_individual_coeff()
        if pvals is not None:
            pvalues.extend(pvals)
            print("pvalues", pvals)
            print(np.mean(pvalues), np.std(pvalues), np.mean(np.array(pvalues) < 0.05))

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



if __name__ == "__main__":
    make_a_plot()
    #make_a_plot_individual_coeff()
