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

from selection.api import randomization, glm_group_lasso, pairs_bootstrap_glm, multiple_queries, discrete_family, projected_langevin, glm_group_lasso_parametric
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix

@register_report(['pvalue', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_condition(ndraw=10000, burnin=2000,
                   scalings=True): 
    s, n, p = 6, 600, 40

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.2, snr=5)
    randomizer = randomization.isotropic_gaussian((p,), scale=sigma)

    lam_frac = 1.5

    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    view = []
    nview = 3
    for i in range(nview):
        view.append(glm_group_lasso(loss, epsilon, penalty, randomizer))

    mv = multiple_queries(view)
    mv.solve()

    active_union = np.zeros(p, np.bool)
    for i in range(nview):
        active_union += view[i].selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)

    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        if nactive==s:
            return None

        active_set = np.nonzero(active_union)[0]
        form_covariances = glm_nonparametric_bootstrap(n, n)
        mv.setup_sampler(form_covariances)

        if scalings == 'tryall': # try condition on some scalings
            view[0].condition_on_scalings()
            view[0].condition_on_subgradient()
            view[1].condition_on_subgradient()
            view[2].condition_on_scalings()
        else:
            view[0].condition_on_subgradient()
            view[1].condition_on_subgradient()
            view[2].condition_on_subgradient()

        target, target_observed = pairs_bootstrap_glm(loss, active_union)
        target_sampler = mv.setup_target(target, target_observed)

        pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                     alternative='twosided',
                                                     ndraw=ndraw,
                                                     burnin=burnin)

        active_var = np.zeros_like(pvalues, np.bool)
        active_var[nonzero] = True
        return pvalues, active_var

def report(niter=50, **kwargs):

    condition_report = reports.reports['test_condition']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pvalue_plot(runs)
    fig.savefig('conditional_pivots.pdf')
