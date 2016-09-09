import numpy as np

from selection.algorithms.lasso import instance as lasso_instance
from selection.algorithms.randomized import randomized_lasso, randomized_logistic
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_for_test

from selection.randomized.tests import logistic_instance, wait_for_return_value

@wait_for_return_value
@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_logistic(n=200, p=30, burnin=2000, ndraw=8000,
                  compute_interval=False,
                  sandwich=True,
                  selected=False,
                  s=6,
                  snr=10,
                  nsim=None):

    X, y, beta, lasso_active = logistic_instance(n=n, p=p, snr=snr, s=s, scale=False, center=False,
                                                 rho=0.1)
    n, p = X.shape

    lam = 0.6 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0)) 

    L = randomized_logistic(y, X, lam, (True, 0.4 * np.diag(np.sqrt(np.diag(np.dot(X.T, X))))),
                            sandwich=sandwich,
                            selected=selected)
    L.fit()

    if (set(range(s)).issubset(L.active) and 
        L.active.shape[0] > s):
        L.unbiased_estimate[:] = np.zeros(p)
        L.constraints.mean[:p] = 0 * L.unbiased_estimate

        v = np.zeros_like(L.active)
        v[s] = 1.
        P0, interval = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                         compute_interval=compute_interval)
        target = (beta[L.active]*v).sum()
        estimate = (L.unbiased_estimate[:L.active.shape[0]]*v).sum()
        low, hi = interval

        v = np.zeros_like(L.active)
        v[0] = 1.
        PA, _ = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                  compute_interval=compute_interval)

        return P0, PA, L

@wait_for_return_value
@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_gaussian(n=200, p=30, burnin=2000, ndraw=8000,
                  compute_interval=False,
                  sandwich=True,
                  selected=False,
                  s=6,
                  snr=7,
                  nsim=None):

    X, y, beta, lasso_active, sigma = lasso_instance(n=n, 
                                                     p=p,
                                                     snr=snr,
                                                     s=s,
                                                     rho=0.1)
    n, p = X.shape

    lam = 2. * sigma * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    L = randomized_lasso(y, X, lam, (True, 0.8 * sigma * np.diag(np.sqrt(np.diag(np.dot(X.T, X))))),
                         sandwich=sandwich,
                         selected=selected,
                         dispersion=sigma**2)

    L.fit()
    if (set(range(s)).issubset(L.active) and 
        L.active.shape[0] > s):
        L.unbiased_estimate[:] = np.zeros(p)
        L.constraints.mean[:p] = 0 * L.unbiased_estimate

        v = np.zeros_like(L.active)
        v[s] = 1.
        P0, interval = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                         compute_interval=compute_interval)
        target = (beta[L.active]*v).sum()
        estimate = (L.unbiased_estimate[:L.active.shape[0]]*v).sum()
        low, hi = interval

        v = np.zeros_like(L.active)
        v[0] = 1.
        PA, _ = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                  compute_interval=compute_interval)

        return P0, PA, L

def compare_sandwich(selected=False, min_sim=500,
                     n=500,
                     p=50,
                     s=5,
                     snr=10,
                     logistic=True,
                     condition_on_more=False):

    P0 = {}
    PA = {}

    def nanclean(v, remove_zeros=False):
        v = np.asarray(v)
        v = v[~np.isnan(v)]
        if remove_zeros:
            return v[v>1.e-6]
        return v

    def nonnan(v):
        v = np.asarray(v)
        return (~np.isnan(v)).sum()

    counter = 0
    no_except = 0
    for i in range(2000):
        for sandwich in [True,False]:
            P0.setdefault(sandwich, [])
            PA.setdefault(sandwich, [])
            print(selected, 'selected')
            try:
                if logistic:
                    R = test_logistic(burnin=2000, ndraw=8000, sandwich=sandwich, selected=selected,
                                      n=n, p=p, s=s, snr=snr, condition_on_more=condition_on_more)
                else:
                    R = test_gaussian(burnin=2000, ndraw=8000, sandwich=sandwich, selected=selected,
                                      n=n, p=p, s=s, snr=snr, condition_on_more=condition_on_more)
                no_except += 1
                if R is not None:
                    P0[sandwich].append(R[0])
                    PA[sandwich].append(R[1])
                    counter += 1
                    print(counter * 1. / no_except, 'screen')
            except np.linalg.LinAlgError:
                pass
        if ((nonnan(P0[True]) > min_sim)
            and (nonnan(P0[False]) > min_sim)
            and (nonnan(PA[False]) > min_sim)
            and (nonnan(PA[True]) > min_sim)):
            break
        print(nonnan(P0[True]), nonnan(P0[False]), nonnan(PA[True]), nonnan(PA[False]))
        print(np.mean(nanclean(P0[True], remove_zeros=True)), np.std(nanclean(P0[True], remove_zeros=True)), 'sandwich')
        print(np.mean(nanclean(P0[False], remove_zeros=True)), np.std(nanclean(P0[False], remove_zeros=True)), 'parametric')

        if i % 25 == 0 and i > 20:
            # make any plots not use display

            from matplotlib import use
            use('Agg')
            import matplotlib.pyplot as plt

            # used for ECDF

            import statsmodels.api as sm


            plt.clf()
            U = np.linspace(0,1, 101)

            plt.plot(U, sm.distributions.ECDF(nanclean(PA[False]))(U), 'r--', label='Parametric', linewidth=3)
            plt.plot(U, sm.distributions.ECDF(nanclean(PA[True]))(U), 'r:', label='Sandwich', linewidth=3)
            plt.plot(U, sm.distributions.ECDF(nanclean(P0[False], remove_zeros=True))(U), 'k--', linewidth=3)
            plt.plot(U, sm.distributions.ECDF(nanclean(P0[True], remove_zeros=True))(U), 'k:', linewidth=3)
            plt.legend(loc='lower right')
            if selected:
                plt.savefig('compare_selected.pdf')
            else:
                plt.savefig('compare_saturated.pdf')


    
