import numpy as np
import numpy as np

import regreg.api as rr

from selection.api import randomization, glm_group_lasso, pairs_bootstrap_glm, multiple_views, discrete_family, projected_langevin
from selection.randomized.tests import logistic_instance
from selection.randomized.M_estimator import restricted_Mest

from selection.randomized.intervals.estimation import estimation, instance


class intervals():

    def __init__(self):
        self.grid_length = 400
        self.param_values_at_grid = np.linspace(-10, 10, num=self.grid_length)


    def setup_samples(self, ref_vec, samples, observed, variances):
        (self.ref_vec,
         self.samples,
         self.observed,
         self.variances) = (ref_vec,
                            samples,
                            observed,
                            variances)

        self.nactive = ref_vec.shape[0]
        self.nsamples = self.samples.shape[1]


    def empirical_exp(self, j, param):
        ref = self.ref_vec[j]
        factor = np.true_divide(param-ref, self.variances[j])
        tilted_samples = np.exp(self.samples[j,:]*factor)
        return np.sum(tilted_samples)/float(self.nsamples)


    def pvalue_by_tilting(self, j, param):
        ref = self.ref_vec[j]
        indicator = np.array(self.samples[j,:] < self.observed[j], dtype =int)
        log_gaussian_tilt = np.array(self.samples[j,:]) * (param - ref)
        log_gaussian_tilt /= self.variances[j]
        emp_exp = self.empirical_exp(j, param)
        LR = np.true_divide(np.exp(log_gaussian_tilt), emp_exp)
        return np.clip(np.sum(np.multiply(indicator, LR)) / float(self.nsamples), 0, 1)


    def pvalues_param_all(self, param_vec):
        pvalues = []
        for j in range(self.nactive):
            pval = self.pvalue_by_tilting(j, param_vec[j])
            pval = 2 * min(pval, 1 - pval)
            pvalues.append(pval)
        return pvalues

    def pvalues_ref_all(self):
        pvalues = []
        for j in range(self.nactive):
            indicator = np.array(self.samples[j, :] < self.observed[j], dtype=int)
            pval = np.sum(indicator)/float(self.nsamples)
            pval = 2*min(pval, 1-pval)
            pvalues.append(pval)

        return pvalues


    def pvalues_grid(self, j):
        pvalues_at_grid = [self.pvalue_by_tilting(j, self.param_values_at_grid[i]) for i in range(self.grid_length)]
        pvalues_at_grid = [2*min(pval, 1-pval) for pval in pvalues_at_grid]
        pvalues_at_grid = np.asarray(pvalues_at_grid, dtype=np.float32)
        return pvalues_at_grid


    def construct_intervals(self, j, alpha=0.1):
        pvalues_at_grid = self.pvalues_grid(j)
        accepted_indices = np.array(pvalues_at_grid > alpha)
        #accepted_indices = np.multiply(accepted_indices, np.array(pvalues_at_grid<1-alpha))
        if np.sum(accepted_indices)>0:
            self.L = np.min(self.param_values_at_grid[accepted_indices])
            self.U = np.max(self.param_values_at_grid[accepted_indices])
            return self.L, self.U

    def construct_intervals_all(self, truth_vec, alpha=0.1):
        ncovered = 0
        nparam = 0
        for j in range(self.nactive):
            LU = self.construct_intervals(j, alpha=alpha)
            if LU is not None:
                L, U = LU
                print "interval", L, U
                nparam +=1
                if (L <= truth_vec[j]) and (U >= truth_vec[j]):
                     ncovered +=1
        return ncovered, nparam


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
        sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        mv.setup_sampler(sampler)
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