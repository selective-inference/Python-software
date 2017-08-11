import numpy as np, pandas as pd
import nose.tools as nt
import numpy.testing.decorators as dec
from itertools import product

from selection.tests.flags import SMALL_SAMPLES
from selection.tests.instance import (gaussian_instance as instance,
                                      logistic_instance)
from selection.tests.decorators import set_sampling_params_iftrue, wait_for_return_value, register_report
import selection.tests.reports as reports

from selection.algorithms.lasso import (lasso, 
                                        data_carving, 
                                        data_splitting,
                                        split_model, 
                                        standard_lasso,
                                        nominal_intervals,
                                        glm_sandwich_estimator,
                                        glm_parametric_estimator)
from selection.algorithms.sqrt_lasso import (solve_sqrt_lasso, choose_lambda)

import regreg.api as rr


try:
    import statsmodels.api
    statsmodels_available = True
except ImportError:
    statsmodels_available = False

def test_gaussian(n=100, p=20):

    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))
    Q = rr.identity_quadratic(0.01, 0, np.ones(p), 0)

    weights_with_zeros = 0.5*lam_theor * np.ones(p)
    weights_with_zeros[:3] = 0.

    huge_weights = weights_with_zeros * 10000

    for q, fw in product([Q, None],
                         [0.5*lam_theor, weights_with_zeros, huge_weights]):

        L = lasso.gaussian(X, y, fw, 1., quadratic=Q)
        L.fit()
        C = L.constraints

        sandwich = glm_sandwich_estimator(L.loglike, B=5000)
        L = lasso.gaussian(X, y, fw, 1., quadratic=Q, covariance_estimator=sandwich)
        L.fit()
        C = L.constraints

        S = L.summary('onesided', compute_intervals=True)
        S = L.summary('twosided')

        nt.assert_raises(ValueError, L.summary, 'none')
        print(L.active)
        yield (np.testing.assert_array_less,
               np.dot(L.constraints.linear_part, L.onestep_estimator),
               L.constraints.offset)

def test_sqrt_lasso(n=100, p=20):

    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0)) / np.sqrt(n)
    Q = rr.identity_quadratic(0.01, 0, np.random.standard_normal(p) / 5., 0)

    weights_with_zeros = 0.5*lam_theor * np.ones(p)
    weights_with_zeros[:3] = 0.

    huge_weights = weights_with_zeros * 10000

    for q, fw in product([None, Q],
                         [0.5*lam_theor, weights_with_zeros, huge_weights]):

        L = lasso.sqrt_lasso(X, y, fw, quadratic=q, solve_args={'min_its':300, 'tol':1.e-12})
        L.fit(solve_args={'min_its':300, 'tol':1.e-12})
        C = L.constraints

        S = L.summary('onesided', compute_intervals=True)
        S = L.summary('twosided')

        yield (np.testing.assert_array_less,
               np.dot(L.constraints.linear_part, L.onestep_estimator),
               L.constraints.offset)


def test_logistic():

    for Y, T in [(np.random.binomial(1,0.5,size=(10,)),
                  np.ones(10)),
                 (np.random.binomial(1,0.5,size=(10,)),
                  None),
                 (np.random.binomial(3,0.5,size=(10,)),
                  3*np.ones(10))]:
        X = np.random.standard_normal((10,5))

        L = lasso.logistic(X, Y, 0.1, trials=T)
        L.fit()

        L = lasso.logistic(X, Y, 0.1, trials=T)
        L.fit()

        C = L.constraints

        np.testing.assert_array_less( \
            np.dot(L.constraints.linear_part, L.onestep_estimator),
            L.constraints.offset)

        P = L.summary()['pval']

        return L, C, P

def test_poisson():

    X = np.random.standard_normal((10,5))
    Y = np.random.poisson(10, size=(10,))

    L = lasso.poisson(X, Y, 0.1)
    L.fit()

    L = lasso.poisson(X, Y, 0.1)
    L.fit()

    C = L.constraints

    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L.onestep_estimator),
        L.constraints.offset)

    P = L.summary()['pval']

    return L, C, P

@dec.skipif(not statsmodels_available, "needs statsmodels")
def test_coxph():

    Q = rr.identity_quadratic(0.01, 0, np.ones(5), 0)
    X = np.random.standard_normal((100,5))
    T = np.random.standard_exponential(100)
    S = np.random.binomial(1, 0.5, size=(100,))

    L = lasso.coxph(X, T, S, 0.1, quadratic=Q)
    L.fit()

    L = lasso.coxph(X, T, S, 0.1, quadratic=Q)
    L.fit()

    C = L.constraints

    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L.onestep_estimator),
        L.constraints.offset)

    P = L.summary()['pval']

    return L, C, P

@register_report(['pvalue', 'split_pvalue', 'active'])
@wait_for_return_value(max_tries=100)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_data_carving_gaussian(n=200,
                               p=200,
                               s=7,
                               sigma=5,
                               rho=0.3,
                               signal=7.,
                               split_frac=0.8,
                               lam_frac=2.,
                               ndraw=8000,
                               burnin=2000, 
                               df=np.inf,
                               compute_intervals=True,
                               use_full_cov=True,
                               return_only_screening=True):

    X, y, beta, true_active, sigma = instance(n=n, 
                                              p=p, 
                                              s=s, 
                                              sigma=sigma, 
                                              rho=rho, 
                                              signal=signal, 
                                              df=df)
    mu = np.dot(X, beta)

    idx = np.arange(n)
    np.random.shuffle(idx)
    stage_one = idx[:int(n*split_frac)]

    lam_theor = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 5000)))).max(0)) * sigma
    DC = data_carving.gaussian(X, y, feature_weights=lam_theor,
                               sigma=sigma,
                               stage_one=stage_one)
    DC.fit()

    if len(DC.active) < n - int(n*split_frac):
        DS = data_splitting.gaussian(X, y, feature_weights=lam_theor,
                                     sigma=sigma,
                                     stage_one=stage_one)
        DS.fit(use_full_cov=True)
        DS.fit(use_full_cov=False)
        DS.fit(use_full_cov=use_full_cov)
        data_split = True
    else:
        print('not enough data for second stage data splitting')
        print(DC.active)
        data_split = False

    if set(true_active).issubset(DC.active):
        carve = []
        split = []
        for var in DC.active:
            carve.append(DC.hypothesis_test(var, burnin=burnin, ndraw=ndraw))
            if data_split:
                split.append(DS.hypothesis_test(var))
            else:
                split.append(np.random.sample()) # appropriate p-value if data splitting can't estimate 2nd stage

        Xa = X[:,DC.active]
        truth = np.dot(np.linalg.pinv(Xa), mu) 

        active = np.zeros(p, np.bool)
        active[true_active] = 1
        v = (carve, split, active)
        return v

@register_report(['pvalue', 'split_pvalue', 'active'])
@wait_for_return_value()
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_data_carving_sqrt_lasso(n=200,
                                 p=200,
                                 s=7,
                                 sigma=5,
                                 rho=0.3,
                                 signal=7.,
                                 split_frac=0.9,
                                 lam_frac=1.2,
                                 ndraw=8000,
                                 burnin=2000, 
                                 df=np.inf,
                                 compute_intervals=True,
                                 return_only_screening=True):
    
    X, y, beta, true_active, sigma = instance(n=n, 
                                         p=p, 
                                         s=s, 
                                         sigma=sigma, 
                                         rho=rho, 
                                         signal=signal, 
                                         df=df)
    mu = np.dot(X, beta)

    idx = np.arange(n)
    np.random.shuffle(idx)
    stage_one = idx[:int(n*split_frac)]
    n1 = len(stage_one)

    lam_theor = lam_frac * np.mean(np.fabs(np.dot(X[stage_one].T, np.random.standard_normal((n1, 5000)))).max(0)) / np.sqrt(n1)
    DC = data_carving.sqrt_lasso(X, y, feature_weights=lam_theor,
                                 stage_one=stage_one)

    DC.fit()

    if len(DC.active) < n - int(n*split_frac):
        DS = data_splitting.sqrt_lasso(X, y, feature_weights=lam_theor,
                                       stage_one=stage_one)
        DS.fit(use_full_cov=True)
        data_split = True
    else:
        print('not enough data for second stage data splitting')
        print(DC.active)
        data_split = False

    if set(true_active).issubset(DC.active):
        carve = []
        split = []
        for var in DC.active:
            carve.append(DC.hypothesis_test(var, burnin=burnin, ndraw=ndraw))
            if data_split:
                split.append(DS.hypothesis_test(var))
            else:
                split.append(np.random.sample())
                

        Xa = X[:,DC.active]
        truth = np.dot(np.linalg.pinv(Xa), mu) 

        active = np.zeros(p, np.bool)
        active[true_active] = 1
        v = (carve, split, active)
        return v


@register_report(['pvalue', 'split_pvalue', 'active'])
@wait_for_return_value()
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_data_carving_logistic(n=700,
                               p=300,
                               s=5,
                               rho=0.05,
                               signal=12.,
                               split_frac=0.8,
                               ndraw=8000,
                               burnin=2000, 
                               df=np.inf,
                               compute_intervals=True,
                               use_full_cov=False,
                               return_only_screening=True):
    
    X, y, beta, true_active = logistic_instance(n=n, 
                                                p=p, 
                                                s=s, 
                                                rho=rho, 
                                                signal=signal,
                                                equicorrelated=False)

    mu = X.dot(beta)
    prob = np.exp(mu) / (1 + np.exp(mu))

    X = np.hstack([np.ones((n,1)), X])
    active = np.array(true_active)
    active += 1
    s += 1
    active = [0] + list(active)
    true_active = active

    idx = np.arange(n)
    np.random.shuffle(idx)
    stage_one = idx[:int(n*split_frac)]
    n1 = len(stage_one)

    lam_theor = 1.0 * np.ones(p+1)
    lam_theor[0] = 0.
    DC = data_carving.logistic(X, y, 
                               feature_weights=lam_theor,
                               stage_one=stage_one)

    DC.fit()

    if len(DC.active) < n - int(n*split_frac):
        DS = data_splitting.logistic(X, y, feature_weights=lam_theor,
                                     stage_one=stage_one)
        DS.fit(use_full_cov=True)
        data_split = True
    else:
        print('not enough data for data splitting second stage')
        print(DC.active)
        data_split = False

    print(true_active, DC.active)
    if set(true_active).issubset(DC.active):
        carve = []
        split = []
        for var in DC.active:
            carve.append(DC.hypothesis_test(var, burnin=burnin, ndraw=ndraw))
            if data_split:
                split.append(DS.hypothesis_test(var))
            else:
                split.append(np.random.sample())

        Xa = X[:,DC.active]

        active = np.zeros(p, np.bool)
        active[true_active] = 1
        v = (carve, split, active)
        return v

@register_report(['pvalue', 'split_pvalue', 'active'])
@wait_for_return_value()
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_data_carving_poisson(n=500,
                              p=300,
                              s=5,
                              sigma=5,
                              rho=0.3,
                              signal=12.,
                              split_frac=0.8,
                              lam_frac=1.2,
                              ndraw=8000,
                              burnin=2000, 
                              df=np.inf,
                              compute_intervals=True,
                              use_full_cov=True,
                              return_only_screening=True):
    
    X, y, beta, true_active, sigma = instance(n=n, 
                                              p=p, 
                                              s=s, 
                                              sigma=sigma, 
                                              rho=rho, 
                                              signal=signal, 
                                              df=df)
    X = np.hstack([np.ones((n,1)), X])
    y = np.random.poisson(10, size=y.shape)
    s = 1
    true_active = [0]

    idx = np.arange(n)
    np.random.shuffle(idx)
    stage_one = idx[:int(n*split_frac)]
    n1 = len(stage_one)

    lam_theor = 3. * np.ones(p+1)
    lam_theor[0] = 0.
    DC = data_carving.poisson(X, y, feature_weights=lam_theor,
                              stage_one=stage_one)

    DC.fit()

    if len(DC.active) < n - int(n*split_frac):
        DS = data_splitting.poisson(X, y, feature_weights=lam_theor,
                                     stage_one=stage_one)
        DS.fit(use_full_cov=True)
        data_split = True
    else:
        print('not enough data for data splitting second stage')
        print(DC.active)
        data_split = False

    print(DC.active)
    if set(true_active).issubset(DC.active):
        carve = []
        split = []
        for var in DC.active:
            carve.append(DC.hypothesis_test(var, burnin=burnin, ndraw=ndraw))
            if data_split:
                split.append(DS.hypothesis_test(var))
            else:
                split.append(np.random.sample())

        Xa = X[:,DC.active]

        active = np.zeros(p, np.bool)
        active[true_active] = 1
        v = (carve, split, active)
        return v
       


@register_report(['pvalue', 'split_pvalue', 'active'])
@wait_for_return_value()
@dec.skipif(not statsmodels_available, "needs statsmodels")
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_data_carving_coxph(n=400,
                            p=20,
                            split_frac=0.8,
                            lam_frac=1.2,
                            ndraw=8000,
                            burnin=2000, 
                            df=np.inf,
                            compute_intervals=True,
                            return_only_screening=True):
  

    X = np.random.standard_normal((n,p))
    T = np.random.standard_exponential(n)
    S = np.random.binomial(1, 0.5, size=(n,))

    true_active = []
    s = 0
    active = np.array(true_active)

    idx = np.arange(n)
    np.random.shuffle(idx)
    stage_one = idx[:int(n*split_frac)]
    n1 = len(stage_one)

    lam_theor = 10. * np.ones(p)
    lam_theor[0] = 0.
    DC = data_carving.coxph(X, T, S, feature_weights=lam_theor,
                            stage_one=stage_one)

    DC.fit()

    if len(DC.active) < n - int(n*split_frac):
        DS = data_splitting.coxph(X, T, S, feature_weights=lam_theor,
                                     stage_one=stage_one)
        DS.fit(use_full_cov=True)
        data_split = True
    else:
        print('not enough data for data splitting second stage')
        print(DC.active)
        data_split = False

    if set(true_active).issubset(DC.active):
        carve = []
        split = []
        for var in DC.active:
            carve.append(DC.hypothesis_test(var, burnin=burnin, ndraw=ndraw))
            if data_split:
                split.append(DS.hypothesis_test(var))
            else:
                split.append(np.random.sample())

        Xa = X[:,DC.active]

        active = np.zeros(p, np.bool)
        active[true_active] = 1
        v = (carve, split, active)
        return v

def test_intervals(n=100, p=20, s=5):
    t = []
    X, y, beta, true_active, sigma = instance(n=n, p=p, s=s)
    las = lasso.gaussian(X, y, 4., sigma=sigma)
    las.fit()

    # smoke test

    las.soln
    las.constraints
    S = las.summary(compute_intervals=True)
    nominal_intervals(las)
    
@register_report(['pvalue', 'active'])
@wait_for_return_value()
def test_gaussian_pvals(n=100,
                        p=500,
                        s=7,
                        sigma=5,
                        rho=0.3,
                        signal=8.):

    X, y, beta, true_active, sigma = instance(n=n, 
                                         p=p, 
                                         s=s, 
                                         sigma=sigma, 
                                         rho=rho, 
                                         signal=signal)
    L = lasso.gaussian(X, y, 20., sigma=sigma)
    L.fit()
    L.fit(L.lasso_solution)
    if set(true_active).issubset(L.active):
        S = L.summary('onesided')
        S = L.summary('twosided')
        return S['pval'], [v in true_active for v in S['variable']]

@register_report(['pvalue', 'active'])
@wait_for_return_value()
def test_sqrt_lasso_pvals(n=100,
                          p=200,
                          s=7,
                          sigma=5,
                          rho=0.3,
                          signal=7.):

    X, y, beta, true_active, sigma = instance(n=n, 
                                         p=p, 
                                         s=s, 
                                         sigma=sigma, 
                                         rho=rho, 
                                         signal=signal)

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0)) / np.sqrt(n)
    Q = rr.identity_quadratic(0.01, 0, np.ones(p), 0)

    weights_with_zeros = 0.7*lam_theor * np.ones(p)
    weights_with_zeros[:3] = 0.

    lasso.sqrt_lasso(X, y, weights_with_zeros, covariance='parametric')
    L = lasso.sqrt_lasso(X, y, weights_with_zeros)
    L.fit()
    if set(true_active).issubset(L.active):
        S = L.summary('onesided')
        S = L.summary('twosided')
        return S['pval'], [v in true_active for v in S['variable']]


@register_report(['pvalue', 'active'])
@wait_for_return_value()
def test_sqrt_lasso_sandwich_pvals(n=200,
                                   p=50,
                                   s=10,
                                   sigma=10,
                                   rho=0.3,
                                   signal=6.,
                                   use_lasso_sd=False):

    X, y, beta, true_active, sigma = instance(n=n, 
                                         p=p, 
                                         s=s, 
                                         sigma=sigma, 
                                         rho=rho, 
                                         signal=signal)

    heteroscedastic_error = sigma * np.random.standard_normal(n) * (np.fabs(X[:,-1]) + 0.5)**2
    heteroscedastic_error += sigma * np.random.standard_normal(n) * (np.fabs(X[:,-2]) + 0.2)**2
    heteroscedastic_error += sigma * np.random.standard_normal(n) * (np.fabs(X[:,-3]) + 0.5)**2
    y += heteroscedastic_error

    feature_weights = np.ones(p) * choose_lambda(X)
    feature_weights[10:12] = 0

    L_SQ = lasso.sqrt_lasso(X, y, feature_weights, covariance='sandwich')
    L_SQ.fit()

    if set(true_active).issubset(L_SQ.active):
        S = L_SQ.summary('twosided')
        return S['pval'], [v in true_active for v in S['variable']]

@register_report(['pvalue', 'parametric_pvalue', 'active'])
@wait_for_return_value()
def test_gaussian_sandwich_pvals(n=200,
                                 p=50,
                                 s=10,
                                 sigma=10,
                                 rho=0.3,
                                 signal=6.,
                                 use_lasso_sd=False):

    X, y, beta, true_active, sigma = instance(n=n, 
                                         p=p, 
                                         s=s, 
                                         sigma=sigma, 
                                         rho=rho, 
                                         signal=signal)

    heteroscedastic_error = sigma * np.random.standard_normal(n) * (np.fabs(X[:,-1]) + 0.5)**2
    heteroscedastic_error += sigma * np.random.standard_normal(n) * (np.fabs(X[:,-2]) + 0.2)**2
    heteroscedastic_error += sigma * np.random.standard_normal(n) * (np.fabs(X[:,-3]) + 0.5)**2
    y += heteroscedastic_error

    # two different estimators of variance
    loss = rr.glm.gaussian(X, y)
    sandwich = glm_sandwich_estimator(loss, B=5000)


    # make sure things work with some unpenalized columns

    feature_weights = np.ones(p) * 3 * sigma
    feature_weights[10:12] = 0

    # try using RSS from LASSO to estimate sigma 

    if use_lasso_sd:
        L_prelim = lasso.gaussian(X, y, feature_weights)
        L_prelim.fit()
        beta_lasso = L_prelim.lasso_solution
        sigma_hat = np.linalg.norm(y - X.dot(beta_lasso))**2 / (n - len(L_prelim.active))
        parametric = glm_parametric_estimator(loss, dispersion=sigma_hat**2)
    else:
        parametric = glm_parametric_estimator(loss, dispersion=None)

    L_P = lasso.gaussian(X, y, feature_weights, covariance_estimator=parametric)
    L_P.fit()

    if set(true_active).issubset(L_P.active):

        S = L_P.summary('twosided')
        P_P = [p for p, v in zip(S['pval'], S['variable']) if v not in true_active]

        L_S = lasso.gaussian(X, y, feature_weights, covariance_estimator=sandwich)
        L_S.fit()

        S = L_S.summary('twosided')
        P_S = [p for p, v in zip(S['pval'], S['variable']) if v not in true_active]

        return P_P, P_S, [v in true_active for v in S['variable']]


@register_report(['pvalue', 'active'])
@wait_for_return_value()
def test_logistic_pvals(n=500,
                        p=200,
                        s=3,
                        rho=0.3,
                        signal=15.):

    X, y, beta, true_active = logistic_instance(n=n, 
                                                p=p, 
                                                s=s, 
                                                rho=rho, 
                                                signal=signal,
                                                equicorrelated=False)

    X = np.hstack([np.ones((n,1)), X])

    print(true_active, 'true')
    active = np.array(true_active)
    active += 1
    active = [0] + list(active)
    true_active = active

    L = lasso.logistic(X, y, [0]*1 + [1.2]*p)
    L.fit()
    S = L.summary('onesided')

    print(true_active, L.active)
    if set(true_active).issubset(L.active):
        return S['pval'], [v in true_active for v in S['variable']]

def test_adding_quadratic_lasso():

    X, y, beta, true_active, sigma = instance(n=300, p=200)
    Q = rr.identity_quadratic(0.01, 0, np.random.standard_normal(X.shape[1]), 0)

    L1 = lasso.gaussian(X, y, 20, quadratic=Q)
    beta1 = L1.fit(solve_args={'min_its':500, 'tol':1.e-12})
    G1 = X[:,L1.active].T.dot(X.dot(beta1) - y) + Q.objective(beta1,'grad')[L1.active]
    np.testing.assert_allclose(G1 * np.sign(beta1[L1.active]), -20)

    lin = rr.identity_quadratic(0.0, 0, np.random.standard_normal(X.shape[1]), 0)
    L2 = lasso.gaussian(X, y, 20, quadratic=lin)
    beta2 = L2.fit(solve_args={'min_its':500, 'tol':1.e-12})
    G2 = X[:,L2.active].T.dot(X.dot(beta2) - y) + lin.objective(beta2,'grad')[L2.active]
    np.testing.assert_allclose(G2 * np.sign(beta2[L2.active]), -20)

def test_equivalence_sqrtlasso(n=200, p=400, s=10, sigma=3.):

    """
    Check equivalent LASSO and sqrtLASSO solutions.
    """

    Y = np.random.standard_normal(n) * sigma
    beta = np.zeros(p)
    beta[:s] = 8 * (2 * np.random.binomial(1, 0.5, size=(s,)) - 1)
    X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    Y += np.dot(X, beta) * sigma
    lam_theor = choose_lambda(X, quantile=0.9)

    weights = lam_theor*np.ones(p)
    weights[:3] = 0.
    soln1, loss1 = solve_sqrt_lasso(X, Y, weights=weights, quadratic=None, solve_args={'min_its':500, 'tol':1.e-10})

    G1 = loss1.smooth_objective(soln1, 'grad') 

    # find active set, and estimate of sigma                                                                                                                          

    active = (soln1 != 0)
    nactive = active.sum()
    subgrad = np.sign(soln1[active]) * weights[active]
    X_E = X[:,active]
    X_Ei = np.linalg.pinv(X_E)
    sigma_E= np.linalg.norm(Y - X_E.dot(X_Ei.dot(Y))) / np.sqrt(n - nactive)

    multiplier = sigma_E * np.sqrt((n - nactive) / (1 - np.linalg.norm(X_Ei.T.dot(subgrad))**2))

    # XXX how should quadratic be changed?                                                                                                                            
    # multiply everything by sigma_E?                                                                                                                                 

    loss2 = rr.glm.gaussian(X, Y)
    penalty = rr.weighted_l1norm(weights, lagrange=multiplier)
    problem = rr.simple_problem(loss2, penalty)

    soln2 = problem.solve(tol=1.e-12, min_its=200)
    G2 = loss2.smooth_objective(soln2, 'grad') / multiplier

    np.testing.assert_allclose(G1[3:], G2[3:])
    np.testing.assert_allclose(soln1, soln2)
    
def report(niter=50, **kwargs):

    # these are all our null tests
    fn_names = ['test_gaussian_pvals',
                'test_logistic_pvals',
                'test_data_carving_gaussian',
                'test_data_carving_sqrt_lasso',
                'test_data_carving_logistic',
                'test_data_carving_poisson',
                'test_data_carving_coxph'
                ]

    dfs = []
    for fn in fn_names:
        fn = reports.reports[fn]
        dfs.append(reports.collect_multiple_runs(fn['test'],
                                                 fn['columns'],
                                                 niter,
                                                 reports.summarize_all))
    dfs = pd.concat(dfs)

    fig = reports.pvalue_plot(dfs)
    fig.savefig('algorithms_pvalues.pdf') 

    fig = reports.split_pvalue_plot(dfs)
    fig.savefig('algorithms_split_pvalues.pdf') 
