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
                           glm_group_lasso_parametric,
                           multiple_queries,
                           glm_target)
from statsmodels.sandbox.stats.multicomp import multipletests
from selection.randomized.cv_view import CV_view


@register_report(['pvalue', 'active_var'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_power(s=30,
               n=2000,
               p=1000,
               rho=0.6,
               equi_correlated=False,
               signal=3.5,
               lam_frac = 1.,
               cross_validation = True,
               condition_on_CVR=True,
               randomizer = 'gaussian',
               randomizer_scale = 1.,
               ndraw=10000,
               burnin=2000,
               loss='gaussian',
               scalings=False,
               subgrad =True,
               parametric=True):

    print(n,p,s)
    if loss=="gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1.,
                                                       equi_correlated=equi_correlated)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        glm_loss = rr.glm.gaussian(X, y)
    elif loss=="logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal, equi_correlated=equi_correlated)
        glm_loss = rr.glm.logistic(X, y)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    if randomizer =='laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer=='gaussian':
        randomizer = randomization.isotropic_gaussian((p,), scale=randomizer_scale)

    epsilon = 1. / np.sqrt(n)

    views = []
    if cross_validation:
        cv = CV_view(glm_loss, loss_label=loss, lasso_randomization=randomizer, epsilon=epsilon,
                     scale1=0.01, scale2=0.01)
        #views.append(cv)
        cv.solve(glmnet=True)
        lam = cv.lam_CVR
        print("minimizer of CVR", lam)

        if condition_on_CVR:
            cv.condition_on_opt_state()
            #lam = np.true_divide(lam+cv.one_SD_rule(direction="up"),2)
            lam = cv.one_SD_rule(direction="up")
            print("one SD rule lambda", lam)


    W = lam_frac * np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

    if parametric == False:
        Mest = glm_group_lasso(glm_loss, epsilon, penalty, randomizer)
    else:
        Mest = glm_group_lasso_parametric(glm_loss, epsilon, penalty, randomizer)

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

    active_set = np.nonzero(active_union)[0]
    print("active set", active_set)
    print("true nonzero", np.nonzero(beta)[0])

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

        target_sampler, target_observed = glm_target(glm_loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False,
                                                     parametric=parametric)
                                                     #reference= beta[active_union])
        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin)
        pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                     parameter=np.zeros_like(target_observed),
                                                     sample=target_sample)
        return pvalues, active_var, s


def BH(pvalues, active_var, s, q=0.2):
    decisions = multipletests(pvalues, alpha=q, method="fdr_bh")[0]
    TP = decisions[active_var].sum()
    FDP = np.true_divide(decisions.sum() - TP, max(decisions.sum(), 1))
    power = np.true_divide(TP, s)
    total_rejections = decisions.sum()
    false_rejections = total_rejections - TP
    return FDP, power, total_rejections, false_rejections

def simple_rejections(pvalues, active_var, s, alpha=0.05):
    decisions = (pvalues < alpha)
    TP = decisions[active_var].sum()
    FDP = np.true_divide(decisions.sum() - TP, max(decisions.sum(), 1))
    nactive = active_var.shape[0]
    FP = np.true_divide(decisions.sum() - TP, nactive)
    power = np.true_divide(TP, s)
    total_rejections = decisions.sum()
    false_rejections = total_rejections - TP
    # selected and survived
    survived = np.true_divide(TP, active_var.sum())
    return FP, FDP, power, total_rejections, false_rejections, nactive, survived


def report(niter=50, **kwargs):
    np.random.seed(500)
    condition_report = reports.reports['test_power']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pivot_plot_simple(runs)
    fig.savefig('marginalized_subgrad_pivots.pdf')


def compute_power(**kwargs):
    BH_sample, simple_rejections_sample = [], []
    niter = 50
    for i in range(niter):
        print("iteration", i)
        result = test_power(**kwargs)[1]
        if result is not None:
            pvalues, active_var, s = result
            BH_sample.append(BH(pvalues, active_var,s))
            simple_rejections_sample.append(simple_rejections(pvalues, active_var,s))

        print("FDP BH mean", np.mean([i[0] for i in BH_sample]))
        print("power BH mean", np.mean([i[1] for i in BH_sample]))
        print("total rejections BH", np.mean([i[2] for i in BH_sample]))
        print("false rejections BH ", np.mean([i[3] for i in BH_sample]))

        print("FP level mean", np.mean([i[0] for i in simple_rejections_sample]))
        print("FDP level mean", np.mean([i[1] for i in simple_rejections_sample]))
        print("power level mean", np.mean([i[2] for i in simple_rejections_sample]))
        print("total rejections level", np.mean([i[3] for i in simple_rejections_sample]))
        print("false rejections level", np.mean([i[4] for i in simple_rejections_sample]))
        print("nactive mean", np.mean([i[5] for i in simple_rejections_sample]))
        print("true variables that survived the second round", np.mean([i[6] for i in simple_rejections_sample]))

    return None


if __name__ == '__main__':
    np.random.seed(500)
    kwargs = {'s':30, 'n':2000, 'p':1000, 'rho':0.6,
              'equi_correlated':False,
              'signal':3.5,
              'lam_frac':1.,
              'cross_validation':True,
              'condition_on_CVR':True,
              'randomizer':'gaussian',
              'randomizer_scale':1.,
              'ndraw':10000,
              'burnin':2000,
              'loss':'gaussian',
              'scalings':False,
              'subgrad':True,
              'parametric':True}
    compute_power(**kwargs)
