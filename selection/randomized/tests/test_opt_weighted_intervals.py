from itertools import product
import numpy as np
import nose.tools as nt

from ..convenience import lasso, step, threshold
from ..query import optimization_sampler
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from ...tests.flags import SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
import matplotlib.pyplot as plt

from scipy.stats import t as tdist
from ..glm import target as glm_target, glm_nonparametric_bootstrap, pairs_bootstrap_glm
from ..M_estimator import restricted_Mest

@set_seed_iftrue(False, 200)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=1000, burnin=100)
def test_opt_weighted_intervals(ndraw=20000, burnin=2000):

    results=[]
    cls = lasso
    for const_info, rand in product(zip([gaussian_instance], [cls.gaussian]), ['laplace', 'gaussian']):

        inst, const = const_info

        X, Y, beta = inst(n=100, p=10, s=3, signal=5.)[:3]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 8
        conv = const(X, Y, W, randomizer=rand)
        signs = conv.fit()
        print("signs", signs)

        #marginalizing_groups = np.zeros(p, np.bool)
        #marginalizing_groups[:int(p/2)] = True
        #conditioning_groups = ~marginalizing_groups
        #conditioning_groups[-int(p/4):] = False

        selected_features = conv._view.selection_variable['variables']

        #if not set(np.where(beta)[0]).issubset(set(np.where(selected_features)[0])):
        #    return None
        #conv.summary(selected_features,
        #             ndraw=ndraw,
        #             burnin=burnin,
        #             compute_intervals=True)

        #conv.decompose_subgradient(marginalizing_groups=marginalizing_groups,
        #                           conditioning_groups=conditioning_groups)

        conv._queries.setup_sampler(form_covariances=None)
        conv._queries.setup_opt_state()
        opt_sampler = optimization_sampler(conv._queries)

        S = opt_sampler.sample(ndraw,
                               burnin,
                               stepsize=1.e-3)
        #print(S.shape)
        #print([np.mean(S[:,i]) for i in range(p)])

        unpenalized_mle = restricted_Mest(conv.loglike, selected_features)
        form_covariances = glm_nonparametric_bootstrap(n, n)
        #conv._queries.setup_sampler(form_covariances)
        boot_target, boot_target_observed = pairs_bootstrap_glm(conv.loglike, selected_features, inactive=None)
        opt_sampler.setup_target(boot_target,
                                 form_covariances)

        sel_pivots = opt_sampler.coefficient_pvalues(unpenalized_mle, parameter = beta[selected_features], sample=S)
        print("pivots ", sel_pivots)
        selective_CI = opt_sampler.confidence_intervals(unpenalized_mle, sample=S)
        print(selective_CI)
        results.append((rand, sel_pivots,selective_CI, beta[selected_features]))

    return results


from statsmodels.distributions import ECDF

def compute_coverage(sel_ci, true_vec):
    nactive = true_vec.shape[0]
    coverage = np.zeros(nactive)
    for i in range(nactive):
        if true_vec[i]>=sel_ci[i,0] and true_vec[i]<=sel_ci[i,1]:
            coverage[i]=1
    return coverage


def main(ndraw=20000, burnin=5000, nsim=10):

    sel_pivots_all = list()
    sel_ci_all = list()
    rand_all = []
    for i in range(nsim):
        for idx, (rand, sel_pivots, sel_ci, true_vec) in enumerate(test_opt_weighted_intervals(ndraw=ndraw, burnin=burnin)):
            if i==0:
                sel_pivots_all.append([])
                rand_all.append(rand)
                sel_ci_all.append([])
            sel_pivots_all[idx].append(sel_pivots)
            sel_ci_all[idx].append(compute_coverage(sel_ci, true_vec))

    xval = np.linspace(0, 1, 200)

    for idx in range(len(rand_all)):
        fig = plt.figure(num=idx, figsize=(8,8))
        plt.clf()
        sel_pivots_all[idx] = [item for sublist in sel_pivots_all[idx] for item in sublist]
        plt.plot(xval, ECDF(sel_pivots_all[idx])(xval), label='selective')
        plt.plot(xval, xval, 'k-', lw=1)
        plt.legend(loc='lower right')

        sel_ci_all[idx] = [item for sublist in sel_ci_all[idx] for item in sublist]
        plt.title(''.join(["coverage ", str(np.mean(sel_ci_all[idx]))]))
        plt.savefig(''.join(["fig", rand_all[idx], '.pdf']))

