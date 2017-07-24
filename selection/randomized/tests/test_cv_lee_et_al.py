import numpy as np
import regreg.api as rr
import pandas as pd
import selection.api as sel
from selection.tests.instance import gaussian_instance
from selection.algorithms.lasso import lasso
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report
from statsmodels.sandbox.stats.multicomp import multipletests
from selection.randomized.cv_view import CV_view
from scipy.stats import norm as ndist
from scipy.optimize import bisect
from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)


def restricted_gaussian(Z, interval=[-5.,5.]):
    L_restrict, U_restrict = interval
    Z_restrict = max(min(Z, U_restrict), L_restrict)
    return ((ndist.cdf(Z_restrict) - ndist.cdf(L_restrict)) /
            (ndist.cdf(U_restrict) - ndist.cdf(L_restrict)))

def pivot(L_constraint, Z, U_constraint, S, truth=0):
    F = restricted_gaussian
    if F((U_constraint - truth) / S) != F((L_constraint -  truth) / S):
        v = ((F((Z-truth)/S) - F((L_constraint-truth)/S)) /
             (F((U_constraint-truth)/S) - F((L_constraint-truth)/S)))
    elif F((U_constraint - truth) / S) < 0.1:
        v = 1
    else:
        v = 0
    return v

def equal_tailed_interval(L_constraint, Z, U_constraint, S, alpha=0.05):

    lb = Z - 5 * S
    ub = Z + 5 * S

    def F(param):
        return pivot(L_constraint, Z, U_constraint, S, truth=param)

    FL = lambda x: (F(x) - (1 - 0.5 * alpha))
    FU = lambda x: (F(x) - 0.5* alpha)
    L_conf = bisect(FL, lb, ub)
    U_conf = bisect(FU, lb, ub)
    return np.array([L_conf, U_conf])


@register_report(['pvalue', 'cover', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive',
                  'active_var','BH_decisions'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_lee_et_al(n=300,
                   p=100,
                   s=10,
                   signal = 3.5,
                   rho = 0.,
                   sigma = 1.,
                   cross_validation=True,
                   condition_on_CVR=False,
                   lam_frac = 0.6,
                   X = None,
                   check_screen=True,
                   intervals=False):

    print(n, p, s)

    if X is None:
        X, y, beta, truth, sigma = gaussian_instance(n=n, p=p, s=s, signal=signal, sigma=sigma, scale=True, center=True)
    else:
        beta = np.zeros(p)
        beta[:s] = signal
        y = X.dot(beta) + np.random.standard_normal(n)*sigma

    truth = np.nonzero(beta != 0)[0]

    if cross_validation:
        cv = CV_view(rr.glm.gaussian(X,y), loss_label="gaussian", lasso_randomization=None, epsilon=None,
                     scale1=None, scale2=None)
        # views.append(cv)
        cv.solve(glmnet=True)
        lam = cv.lam_CVR
        print("minimizer of CVR", lam)

        if condition_on_CVR:
            cv.condition_on_opt_state()
            lam = np.true_divide(lam+cv.one_SD_rule(direction="up"),2)
            #lam = cv.one_SD_rule(direction="up")
            print("one SD rule lambda", lam)
    else:
        lam = lam_frac*np.fabs(X.T.dot(np.random.normal(1, 1. / 2, (n, 1000)))).max()

    L = lasso.gaussian(X, y, lam, sigma=sigma)
    soln = L.fit()

    active = soln != 0
    nactive = active.sum()
    print("nactive", nactive)
    if nactive==0:
        return None

    active_signs = np.sign(soln[active])

    if (check_screen == False) or (set(truth).issubset(np.nonzero(active)[0])):

        active_set = np.nonzero(active)[0]
        print("active set", active_set)
        true_vec = beta[active]
        active_var = np.zeros(nactive, np.bool)

        # Lee et al. using sigma
        pvalues = np.zeros(nactive)
        sel_length = np.zeros(nactive)
        sel_covered = np.zeros(nactive)

        naive_pvalues = np.zeros(nactive)
        naive_length = np.zeros(nactive)
        naive_covered = np.zeros(nactive)

        C = L.constraints

        if C is not None:
            one_step = L.onestep_estimator
            for i in range(one_step.shape[0]):
                eta = np.zeros_like(one_step)
                eta[i] = active_signs[i]
                alpha = 0.1

                def naive_inference():
                    obs = (eta * one_step).sum()
                    sd = np.sqrt(np.dot(eta.T, C.covariance.dot(eta)))
                    Z = obs / sd
                    # use Phi truncated to [-5,5]
                    _pval = ndist.cdf(obs/sigma)
                    _pval = 2 * min(_pval, 1 - _pval)
                    _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                                 obs + ndist.ppf(1 - alpha / 2) * sd)
                    return _pval, _interval

                if C.linear_part.shape[0] > 0:  # there were some constraints
                    L, Z, U, S = C.bounds(eta, one_step)
                    _pval = pivot(L, Z, U, S)
                    # two-sided
                    _pval = 2 * min(_pval, 1 - _pval)

                    if intervals==True:
                        if _pval < 10 ** (-8):
                            return None
                        L, Z, U, S = C.bounds(eta, one_step)
                        _interval = equal_tailed_interval(L, Z, U, S, alpha=alpha)
                        _interval = sorted([_interval[0] * active_signs[i],
                                     _interval[1] * active_signs[i]])
                else:
                    obs = (eta * one_step).sum()
                    ## jelena: should be this sd = np.sqrt(np.dot(eta.T, C.covariance.dot(eta))), no?
                    sd = np.sqrt((eta * C.covariance.dot(eta)))
                    Z = obs / sd
                    _pval = 2 * (ndist.sf(min(np.fabs(Z))) - ndist.sf(5)) / (ndist.cdf(5) - ndist.cdf(-5))
                    if intervals==True:
                        _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                                     obs + ndist.ppf(1 - alpha / 2) * sd)

                pvalues[i] = _pval

                naive_pvalues[i], _naive_interval = naive_inference()

                #print(_naive_interval)

                def coverage(LU):
                    L, U = LU[0], LU[1]
                    _length = U - L
                    _covered = 0
                    if (L <= true_vec[i]) and (U >= true_vec[i]):
                        _covered = 1
                    return _covered, _length

                if intervals==True:
                    sel_covered[i], sel_length[i] = coverage(_interval)
                    naive_covered[i], naive_length[i] = coverage(_naive_interval)

                active_var[i] = active_set[i] in truth
        else:
            return None

        print(pvalues)
        q = 0.2
        BH_desicions = multipletests(pvalues, alpha=q, method="fdr_bh")[0]
        return  pvalues, sel_covered, sel_length, \
                naive_pvalues, naive_covered, naive_length, active_var, BH_desicions


def report(niter=100, design="random", **kwargs):

    if design=="fixed":
        X, _, _, _, _ = gaussian_instance(**kwargs)
        kwargs.update({'X':X})

    intervals_report = reports.reports['test_lee_et_al']
    screened_results = reports.collect_multiple_runs(intervals_report['test'],
                                             intervals_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    screened_results.to_pickle("lee_et_al_pivots.pkl")
    results = pd.read_pickle("lee_et_al_pivots.pkl")

    #naive plus lee et al.
    fig = reports.pivot_plot_plus_naive(results)
    fig.suptitle("Lee et al. and naive p-values", fontsize=20)
    fig.savefig('lee_et_al_pivots.pdf')

    # naive only
    fig1 = reports.naive_pvalue_plot(results)
    fig1.suptitle("Naive p-values", fontsize=20)
    fig1.savefig('naive_pvalues.pdf')


if __name__ == '__main__':

    np.random.seed(500)
    kwargs = {'s': 0, 'n': 500, 'p': 100, 'signal': 3.5, 'sigma': 1, 'rho': 0., 'intervals':False,
              'cross_validation': True, 'condition_on_CVR': False}
    report(niter=100, **kwargs)


