import numpy as np

import regreg.api as rr

from selection.api import randomization, glm_group_lasso, pairs_bootstrap_glm, multiple_views
from selection.randomized.tests import logistic_instance
from selection.randomized.M_estimator import restricted_Mest
from selection.randomized.glm import glm_nonparametric_bootstrap
from selection.randomized.intervals.intervals import intervals

def Langevin_samples(solve_args={'min_its':50, 'tol':1.e-10}):
    s, n, p = 3, 200, 10

    randomizer = randomization.laplace((p,), scale=0.5)
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

    # first randomization
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomizer)
    # second randomization
    #M_est2 = glm_group_lasso(loss, epsilon, penalty, randomizer)

    #mv = multiple_views([M_est1, M_est2])
    mv = multiple_views([M_est1])
    mv.solve()

    #active = M_est1.overall+M_est2.overall
    active = M_est1.overall
    nactive = np.sum(active)
    active_set = np.nonzero(active)[0]
    print "true beta", beta[:s]
    print "active set", active_set
    unpenalized_mle = restricted_Mest(loss, M_est1.overall, solve_args=solve_args)

    nsamples = 10000
    all_samples = np.zeros((nactive, nsamples))
    all_variances = np.zeros(nactive)

    if set(nonzero).issubset(active_set):

        boot_target, target_observed = pairs_bootstrap_glm(loss, active)
        target = lambda indices: boot_target(indices)[:nactive]
        observed = target_observed[:nactive]

        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)
        target_sampler = mv.setup_target(target, observed, reference = unpenalized_mle)
        samples = target_sampler.sample(ndraw = nsamples, burnin = 2000)
        samples  = np.asarray(samples, dtype=np.float32)
        print samples.shape
        for j in range(nactive):
            all_variances[j] = target_sampler.target_cov[j,j]
            all_samples[j,:] = samples[:,j]


        # for j in range(nactive):
        #
        #     boot_target, target_observed = pairs_bootstrap_glm(loss, active)
        #     inactive_target = lambda indices: boot_target(indices)[j]
        #     inactive_observed = target_observed[j]
        #     sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        #
        #     mv.setup_sampler(sampler)
        #     target_sampler = mv.setup_target(inactive_target, inactive_observed, reference = unpenalized_mle[j])
        #
        #     all_variances[j] = target_sampler.target_cov
        #     all_samples[j,:] = target_sampler.sample(ndraw = nsamples, burnin = 2000)

        return beta, active, unpenalized_mle, all_samples, all_variances




def test_intervals():

    Langevin_result = Langevin_samples()

    if Langevin_result is not None:
        beta, active, beta_ref, all_samples, all_variances = Langevin_result
        param_vec = beta[active]

        int_class = intervals()

        int_class.setup_samples(beta_ref, all_samples, beta_ref, all_variances)

        pvalues_ref = int_class.pvalues_ref_all()
        pvalues_param = int_class.pvalues_param_all(param_vec)

        ncovered, nparam = int_class.construct_intervals_all(param_vec)

        #ncovered, nparam = int_class.construct_naive_intervals(param_vec)
        print "pvalue(s) at the truth", pvalues_param
        return np.copy(pvalues_ref), np.copy(pvalues_param), ncovered, nparam



if __name__ == "__main__":

    P_param_all = []
    P_ref_all = []
    P_param_first = []
    P_ref_first = []
    ncovered_total = 0
    nparams_total = 0

    for i in range(100):
        print "\n"
        print "iteration", i
        pvals_ints = test_intervals()
        if pvals_ints is not None:
            # print pvalues
            pvalues_ref, pvalues_param, ncovered, nparam = pvals_ints
            P_ref_all.extend(pvalues_ref)
            P_param_all.extend(pvalues_param)
            P_ref_first.append(pvalues_ref[0])
            P_param_first.append(pvalues_param[0])

            if ncovered is not None:
                ncovered_total += ncovered
                nparams_total += nparam

    print "number of intervals", nparams_total
    print "coverage", ncovered_total/float(nparams_total)



    from matplotlib import pyplot as plt
    import statsmodels.api as sm


    fig = plt.figure()
    plot_pvalues0 = fig.add_subplot(221)
    plot_pvalues1 = fig.add_subplot(222)
    plot_pvalues2 = fig.add_subplot(223)
    plot_pvalues3 = fig.add_subplot(224)


    ecdf = sm.distributions.ECDF(P_param_all)
    x = np.linspace(min(P_param_all), max(P_param_all))
    y = ecdf(x)
    plot_pvalues0.plot(x, y, '-o', lw=2)
    plot_pvalues0.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues0.set_title("P values at the truth")
    plot_pvalues0.set_xlim([0, 1])
    plot_pvalues0.set_ylim([0, 1])


    ecdf1 = sm.distributions.ECDF(P_ref_all)
    x1 = np.linspace(min(P_ref_all), max(P_ref_all))
    y1 = ecdf1(x1)
    plot_pvalues1.plot(x1, y1, '-o', lw=2)
    plot_pvalues1.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues1.set_title("P values at the reference")
    plot_pvalues1.set_xlim([0, 1])
    plot_pvalues1.set_ylim([0, 1])


    ecdf2 = sm.distributions.ECDF(P_ref_first)
    x2 = np.linspace(min(P_ref_first), max(P_ref_first))
    y2 = ecdf2(x2)
    plot_pvalues2.plot(x2, y2, '-o', lw=2)
    plot_pvalues2.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues2.set_title("First P value at the reference")
    plot_pvalues2.set_xlim([0, 1])
    plot_pvalues2.set_ylim([0, 1])


    ecdf3 = sm.distributions.ECDF(P_param_first)
    x3 = np.linspace(min(P_param_first), max(P_param_first))
    y3 = ecdf3(x3)
    plot_pvalues3.plot(x3, y3, '-o', lw=2)
    plot_pvalues3.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues3.set_title("First P value at the truth")
    plot_pvalues3.set_xlim([0, 1])
    plot_pvalues3.set_ylim([0, 1])

    plt.show()
    plt.savefig("P values from the intervals file.png")
