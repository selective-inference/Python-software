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

from selection.randomized.query import naive_confidence_intervals

from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix
from selection.randomized.cv_view import CV_view

@register_report(['pvalue','BH_decisions', 'active_var'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_power(s=10,
               n=1000,
               p=500,
               rho=0.,
               snr=3.5,
               lam_frac = 0.8,
               q = 0.2,
               cross_validation = False,
               ndraw=10000,
               burnin=1000,
               loss='gaussian',
               scalings=False,
               subgrad =True):

    if loss=="gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss=="logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    #randomizer = randomization.laplace((p,), scale=2*sigma)
    randomizer = randomization.isotropic_gaussian((p,), scale=1.)

    epsilon = 1. / np.sqrt(n)

    views = []
    if cross_validation:
        cv = CV_view(loss)
        cv.solve()
        views.append(cv)
        condition_on_CVR = False
        if condition_on_CVR:
            cv.condition_on_opt_state()
        lam = cv.lam_CVR

    W = lam_frac * np.ones(p) * lam
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
    Mest = glm_group_lasso(loss, epsilon, penalty, randomizer)

    views.append(Mest)

    queries = multiple_queries(views)
    queries.solve()

    active_union = np.zeros(p, np.bool)
    active_union += Mest.selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    nonzero = np.where(beta)[0]
    true_vec = beta[active_union]

    check_screen = False
    if check_screen==False:

        if scalings: # try condition on some scalings
             Mest.condition_on_scalings()
        if subgrad:
             Mest.decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool), marginalizing_groups=np.ones(p, bool))

        active_set = np.nonzero(active_union)[0]
        active_var = np.zeros(nactive, np.bool)
        for j in range(nactive):
            active_var[j] = active_set[j] in nonzero

        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False)
                                                     #reference= beta[active_union])
        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin)
        pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                     parameter=np.zeros_like(target_observed),
                                                     sample=target_sample)

        from statsmodels.sandbox.stats.multicomp import multipletests
        BH_decisions = multipletests(pvalues, alpha=q, method="fdr_bh")[0]

        BH_TP = BH_decisions[active_var].sum()
        FDP = np.true_divide(BH_decisions.sum() - BH_TP, max(BH_decisions.sum(), 1))
        power = np.true_divide(BH_TP, s)
        #return pvalues, BH_decisions, active_var # report
        return FDP, power

def report(niter=50, **kwargs):

    condition_report = reports.reports['test_power']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pivot_plot_simple(runs)
    fig.savefig('marginalized_subgrad_pivots.pdf')

def compute_power():
    FDP_sample = []
    power_sample = []
    niter = 50
    for i in range(niter):
        result = test_power()[1]
        if result is not None:
            FDP, power = result
            FDP_sample.append(FDP)
            power_sample.append(power)
        print("FDP mean", np.mean(FDP_sample))
        print("power mean", np.mean(power_sample))
    return FDP_sample, power_sample


def plot_power():
    FDP_yaxis = []
    power_yaxis = []
    n = 500
    niter = 50
    npoints = 5
    p_grid = np.linspace(100, n, npoints)
    for i in range(npoints):
        p = int(p_grid[i])
        FDP_fixed_p, power_fixed_p = [], []
        for j in range(niter):
            print("p, iteration:", p, j)
            result = test_power(n=n, p=p)[1]
            if result is not None:
                FDP, power = result
                FDP_fixed_p.append(FDP)
                power_fixed_p.append(power)
                print("FDP", np.mean(FDP_fixed_p))
                print("power:", np.mean(power_fixed_p))
        FDP_yaxis.append(np.mean(FDP_fixed_p))
        power_yaxis.append(np.mean(power_fixed_p))


    print("FDP sequence:", FDP_yaxis)
    print("power sequence:", power_yaxis)
    return FDP_yaxis, power_yaxis

if __name__ == '__main__':
    compute_power()
