from __future__ import print_function
import numpy as np

import regreg.api as rr
import selection.tests.reports as reports


from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.tests.decorators import (wait_for_return_value, 
                                        set_seed_iftrue, 
                                        set_sampling_params_iftrue,
                                        register_report)
import selection.tests.reports as reports

from selection.api import (randomization, 
                           glm_group_lasso, 
                           pairs_bootstrap_glm, 
                           multiple_queries, 
                           discrete_family, 
                           projected_langevin, 
                           glm_group_lasso_parametric, 
                           glm_target)

from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

@register_report(['truth', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_condition(s=0,
                   n=100,
                   p=200,
                   rho=0.1,
                   snr=10,
                   lam_frac = 1.4,
                   ndraw=10000, burnin=2000,
                   loss='logistic',
                   nviews=1,
                   scalings=False):

    if loss=="gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss=="logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    #randomizer = randomization.isotropic_gaussian((p,), scale=sigma)
    randomizer = randomization.laplace((p,), scale=0.6)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p)*lam
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    views = []
    for i in range(nviews):
        views.append(glm_group_lasso(loss, epsilon, penalty, randomizer))

    queries = multiple_queries(views)
    queries.solve()

    active_union = np.zeros(p, np.bool)
    for view in views:
        active_union += view.selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)

    nonzero = np.where(beta)[0]

    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        if nactive==s:
            return None

        if scalings: # try condition on some scalings
            for i in range(nviews):
                views[i].condition_on_subgradient()
                views[i].condition_on_scalings()
        else:
            for i in range(nviews):
               views[i].condition_on_subgradient()

        active_set = np.nonzero(active_union)[0]
        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries)
                                                     #reference= beta[active_union])
        #print(target_sampler.target_cov)
        test_stat = lambda x: np.linalg.norm(x - beta[active_union])
        observed_test_value = test_stat(target_observed)

        pivots = target_sampler.hypothesis_test(test_stat,
                                               observed_test_value,
                                               alternative='twosided',
                                               parameter = beta[active_union],
                                               ndraw=ndraw,
                                               burnin=burnin)

        return [pivots], [False]

def report(niter=50, **kwargs):

    condition_report = reports.reports['test_condition']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pivot_plot_simple(runs)
    fig.savefig('conditional_pivots.pdf')


if __name__ == '__main__':
    report()