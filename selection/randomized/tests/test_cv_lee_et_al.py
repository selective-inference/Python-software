import functools

import numpy as np
from scipy.stats import norm as ndist
import matplotlib.pyplot as plt

import regreg.api as rr

import selection.api as sel
from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import lasso
from selection.randomized.cv import CV
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
from statsmodels.sandbox.stats.multicomp import multipletests


@register_report(['pvals', 'cover', 'ci_length_clt', 'active_var','BH_decisions'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_lee_et_al(n=3000,
                   p=1000,
                   s=10,
                   snr = 3.5,
                   rho =0.,
                   cross_validation=False,
                   K=5):

    print (n, p, s, rho)

    X, y, beta, truth, sigma = gaussian_instance(n=n, p=p, s=s, snr=snr, sigma=1., scale=True, center=True)

    truth = np.nonzero(beta != 0)[0]

    if cross_validation==True:
        lam_seq = np.exp(np.linspace(np.log(1.e-6), np.log(2), 30)) * np.fabs(X.T.dot(y)).max()
        folds = np.arange(n) % K
        np.random.shuffle(folds)
        CV_compute = CV(rr.glm.gaussian(X,y), folds, lam_seq)
        lam_CV, _, lam_CV_randomized, _ = CV_compute.choose_lambda_CVr()
        lam = lam_CV

    lam = 0.6*np.fabs(X.T.dot(np.random.binomial(1, 1. / 2, (n, 1000)))).max()
    L = lasso.gaussian(X, y, lam)
    soln = L.fit()

    active = soln != 0
    nactive = active.sum()

    if set(truth).issubset(np.nonzero(active)[0]) and active.sum() > 0:

        print("nactive", nactive)
        active_set = np.nonzero(active)[0]
        print("active set", active_set)
        true_vec = beta[active]
        active_var = np.zeros(nactive, np.bool)

        # Lee et al. using sigma
        pvalues = np.zeros(nactive)
        sel_length = np.zeros(nactive)
        sel_covered = np.zeros(nactive)
        L0 = lasso.gaussian(X, y, lam, sigma=sigma)
        L0.fit()

        for i in range(active.sum()):
            active_var[i] = active_set[i] in truth

            keep = np.zeros(active.sum())
            keep[i] = 1.
            pvalues[i] = L0.constraints.pivot(keep,
                                         L.onestep_estimator,
                                         alternative='twosided')
            interval = L0.constraints.interval(keep,
                                    L.onestep_estimator, alpha=0.1)
            sel_length[i] = interval[1] - interval[0]
            if (interval[0] <= true_vec[i]) and (interval[1] >= true_vec[i]):
                    sel_covered[i] = 1

        q = 0.2
        BH_desicions = multipletests(pvalues, alpha=q, method="fdr_bh")[0]
        return  pvalues, sel_covered, sel_length, active_var, BH_desicions


def report(niter=50, **kwargs):

    kwargs = {'s': 10, 'n': 3000, 'p': 1000, 'snr': 3.5}
    intervals_report = reports.reports['test_lee_et_al']
    CV_runs = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)
    #fig = reports.pivot_plot_plus_naive(CV_runs)
    #fig.suptitle("lee et al pivots")
    #fig.savefig('lee et al_pivots.pdf')


def compute_power():
    BH_sample, simple_rejections_sample = [], []
    niter = 50
    for i in range(niter):
        print("iteration", i)
        s = 10
        result = test_lee_et_al(s=s)[1]
        if result is not None:
            pvalues, _, _, active_var, _ = result
            from selection.randomized.tests.test_power import BH, simple_rejections
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
    report()
    #compute_power()