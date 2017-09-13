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

    results = []
    cls = lasso
    for const_info, rand in product(zip([gaussian_instance], [cls.gaussian]), ['laplace', 'gaussian']):

        inst, const = const_info

        X, Y, beta = inst(n=100, p=10, s=0, signal=1., sigma=5.)[:3]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 5
        conv = const(X, Y, W, randomizer=rand, parametric_cov_estimator=True)
        signs = conv.fit()
        print("signs", signs)

        #marginalizing_groups = np.zeros(p, np.bool)
        #marginalizing_groups[:int(p/2)] = True
        #conditioning_groups = ~marginalizing_groups
        #conditioning_groups[-int(p/4):] = False
        #conv.decompose_subgradient(marginalizing_groups=marginalizing_groups,
        #                           conditioning_groups=conditioning_groups)

        selected_features = conv._view.selection_variable['variables']

        sel_pivots, sel_ci = conv.summary(selected_features,
                                          null_value=beta[selected_features],
                                          ndraw=ndraw,
                                          burnin=burnin,
                                          compute_intervals=True)

        results.append((rand, sel_pivots, sel_ci, beta[selected_features]))

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
    np.random.seed(1)

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
            print(sel_ci)
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
        print(sel_ci_all)
        plt.title(''.join(["coverage ", str(np.mean(sel_ci_all[idx]))]))
        plt.savefig(''.join(["fig", rand_all[idx], '.pdf']))

