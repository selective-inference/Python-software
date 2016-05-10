import numpy as np
import numpy.testing.decorators as dec

from itertools import product
from selection.algorithms.lasso import (lasso, 
                                        data_carving, 
                                        data_splitting,
                                        instance, 
                                        split_model, 
                                        standard_lasso,
                                        instance, 
                                        nominal_intervals,
                                        gaussian_sandwich_estimator,
                                        gaussian_parametric_estimator)

from selection.algorithms.sqrt_lasso import solve_sqrt_lasso 

from regreg.api import identity_quadratic

from selection.tests.decorators import set_sampling_params_iftrue

def test_gaussian(n=100, p=20):

    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))
    Q = None # identity_quadratic(0.01, 0, np.ones(p), 0)

    weights_with_zeros = 0.5*lam_theor * np.ones(p)
    weights_with_zeros[:3] = 0.

    for q, fw in product([Q, None],
                         [0.5*lam_theor, weights_with_zeros]):

        L = lasso.gaussian(X, y, fw, 1., quadratic=Q)
        L.fit()
        C = L.constraints

        sandwich = gaussian_sandwich_estimator(X, y)
        L = lasso.gaussian(X, y, fw, 1., quadratic=Q, covariance_estimator=sandwich)
        L.fit()
        C = L.constraints

        S = L.summary('onesided', compute_intervals=True)
        S = L.summary('twosided')

        yield (np.testing.assert_array_less,
               np.dot(L.constraints.linear_part, L.onestep_estimator),
               L.constraints.offset)

def test_sqrt_lasso(n=100, p=20):

    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0)) / np.sqrt(n)
    Q = None # identity_quadratic(0.01, 0, np.random.standard_normal(p) / 5., 0)

    weights_with_zeros = 0.5*lam_theor * np.ones(p)
    weights_with_zeros[:3] = 0.

    for q, fw in product([None, Q],
                         [0.5*lam_theor, weights_with_zeros]):

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

def test_coxph():

    Q = identity_quadratic(0.01, 0, np.ones(5), 0)
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

@set_sampling_params_iftrue(False)
def test_data_carving(n=100,
                      p=200,
                      s=7,
                      sigma=5,
                      rho=0.3,
                      snr=7.,
                      split_frac=0.8,
                      lam_frac=2.,
                      ndraw=8000,
                      burnin=2000, 
                      df=np.inf,
                      coverage=0.90,
                      compute_intervals=True,
                      nsim=None):

    counter = 0

    return_value = []

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr, 
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
            DS.fit()
            data_split = True
        else:
            data_split = False
            print('not enough data for second stage data splitting')

                
        if set(range(s)).issubset(DC.active):
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

            split_coverage = np.nan
            carve_coverage = np.nan

            TP = s
            FP = DC.active.shape[0] - TP
            v = (carve[s:], split[s:], carve[:s], split[:s], counter, carve_coverage, split_coverage, TP, FP)
            return_value.append(v)
            break
        else:
            TP = len(set(DC.active).intersection(range(s)))
            FP = DC.active.shape[0] - TP
            v = (None, None, None, None, counter, np.nan, np.nan, TP, FP)
            return_value.append(v)

    return return_value

@set_sampling_params_iftrue(False)
def test_data_carving_sqrt_lasso(n=100,
                                 p=200,
                                 s=7,
                                 sigma=5,
                                 rho=0.3,
                                 snr=7.,
                                 split_frac=0.9,
                                 lam_frac=1.2,
                                 ndraw=8000,
                                 burnin=2000, 
                                 df=np.inf,
                                 coverage=0.90,
                                 compute_intervals=True,
                                 nsim=None):
    
    counter = 0

    return_value = []

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr, 
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
        DS = data_splitting.sqrt_lasso(X, y, feature_weights=lam_theor,
                                       stage_one=stage_one)
        DS.fit()
                
        if set(range(s)).issubset(DC.active):
            carve = []
            split = []
            for var in DC.active:
                carve.append(DC.hypothesis_test(var, burnin=burnin, ndraw=ndraw))
                split.append(DS.hypothesis_test(var))

            Xa = X[:,DC.active]
            truth = np.dot(np.linalg.pinv(Xa), mu) 

            split_coverage = np.nan
            carve_coverage = np.nan

            TP = s
            FP = DC.active.shape[0] - TP
            v = (carve[s:], split[s:], carve[:s], split[:s], counter, carve_coverage, split_coverage, TP, FP)
            return_value.append(v)
            break
        else:
            TP = len(set(DC.active).intersection(range(s)))
            FP = DC.active.shape[0] - TP
            v = (None, None, None, None, counter, np.nan, np.nan, TP, FP)
            return_value.append(v)

    return return_value

@set_sampling_params_iftrue(False)
def test_data_carving_logistic(n=200,
                               p=300,
                               s=5,
                               sigma=5,
                               rho=0.3,
                               snr=9.,
                               split_frac=0.8,
                               lam_frac=1.2,
                               ndraw=8000,
                               burnin=2000, 
                               df=np.inf,
                               coverage=0.90,
                               compute_intervals=True,
                               nsim=None):
    
    counter = 0

    return_value = []

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr, 
                                             df=df)

        z = (y > 0)
        X = np.hstack([np.ones((n,1)), X])

        active = np.array(active)
        active += 1
        active = [0] + list(active)

        L = lasso.logistic(X, z, [0]*1 + [1.2]*p)

        idx = np.arange(n)
        np.random.shuffle(idx)
        stage_one = idx[:int(n*split_frac)]
        n1 = len(stage_one)

        lam_theor = 1.0 * np.ones(p+1)
        lam_theor[0] = 0.
        DC = data_carving.logistic(X, z, feature_weights=lam_theor,
                                   stage_one=stage_one)

        DC.fit()

        if len(DC.active) < n - int(n*split_frac):
            DS = data_splitting.logistic(X, z, feature_weights=lam_theor,
                                         stage_one=stage_one)
            DS.fit()
            data_split = True
        else:
            print 'not enough data for data splitting second stage'
            data_split = False

        if set(range(s)).issubset(DC.active):
            carve = []
            split = []
            for var in DC.active:
                carve.append(DC.hypothesis_test(var, burnin=burnin, ndraw=ndraw))
                if data_split:
                    split.append(DS.hypothesis_test(var))
                else:
                    split.append(np.random.sample())

            Xa = X[:,DC.active]

            split_coverage = np.nan
            carve_coverage = np.nan

            TP = s
            FP = DC.active.shape[0] - TP
            v = (carve[s:], split[s:], carve[:s], split[:s], counter, carve_coverage, split_coverage, TP, FP)
            return_value.append(v)
            break
        else:
            TP = len(set(DC.active).intersection(range(s)))
            FP = DC.active.shape[0] - TP
            v = (None, None, None, None, counter, np.nan, np.nan, TP, FP)
            return_value.append(v)

    return return_value

@set_sampling_params_iftrue(True)
@dec.skipif(True, "needs a data_carving_coverage function to be defined")
def test_data_carving_coverage(nsim=200, 
                               coverage=0.8,
                               ndraw=8000,
                               burnin=2000):
    C = []
    SE = np.sqrt(coverage * (1 - coverage) / nsim)

    while True:
        C.extend(data_carving_coverage(ndraw=ndraw, burnin=burnin)[-1])
        if len(C) > nsim:
            break

    if np.fabs(np.mean(C) - coverage) > 3 * SE:
        raise ValueError('coverage not within 3 SE of where it should be')

    return C

def test_intervals(n=100, p=20, s=5):
    t = []
    X, y, beta, active, sigma = instance(n=n, p=p, s=s)
    las = lasso.gaussian(X, y, 4., sigma=sigma)
    las.fit()

    # smoke test

    las.soln
    las.constraints
    S = las.summary(compute_intervals=True)
    nominal_intervals(las)
    
def test_gaussian_pvals(n=100,
                        p=500,
                        s=7,
                        sigma=5,
                        rho=0.3,
                        snr=8.):

    counter = 0

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr)
        L = lasso.gaussian(X, y, 20., sigma=sigma)
        L.fit()
        v = {1:'twosided',
             0:'onesided'}[counter % 2]
        if set(active).issubset(L.active):
            S = L.summary(v)
            return [p for p, v in zip(S['pval'], S['variable']) if v not in active]

def test_sqrt_lasso_pvals(n=100,
                          p=200,
                          s=7,
                          sigma=5,
                          rho=0.3,
                          snr=7.):

    counter = 0

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr)

        lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0)) / np.sqrt(n)
        Q = identity_quadratic(0.01, 0, np.ones(p), 0)

        weights_with_zeros = 0.7*lam_theor * np.ones(p)
        weights_with_zeros[:3] = 0.

        L = lasso.sqrt_lasso(X, y, weights_with_zeros)
        L.fit()
        v = {1:'twosided',
             0:'onesided'}[counter % 2]
        if set(active).issubset(L.active):
            S = L.summary(v)
            return [p for p, v in zip(S['pval'], S['variable']) if v not in active]


def test_gaussian_sandwich_pvals(n=100,
                                 p=200,
                                 s=20,
                                 sigma=10,
                                 rho=0.3,
                                 snr=6.):

    counter = 0

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr)

        heteroscedastic_error = sigma * np.random.standard_normal(n) * (np.fabs(X[:,-1]) + 0.5)**2
        heteroscedastic_error += sigma * np.random.standard_normal(n) * (np.fabs(X[:,-2]) + 0.2)**2
        heteroscedastic_error += sigma * np.random.standard_normal(n) * (np.fabs(X[:,-3]) + 0.5)**2
        y += heteroscedastic_error

        # two different estimators of variance
        sandwich = gaussian_sandwich_estimator(X, y, B=1000)
        parametric = gaussian_parametric_estimator(X, y, sigma=None)

        # make sure things work with some unpenalized columns

        feature_weights = np.ones(p) * 3 * sigma
        feature_weights[10:12] = 0
        L_P = lasso.gaussian(X, y, feature_weights, covariance_estimator=parametric)
        L_P.fit()

        if set(active).issubset(L_P.active):

            S = L_P.summary('twosided')
            P_P = [p for p, v in zip(S['pval'], S['variable']) if v not in active]
        
            L_S = lasso.gaussian(X, y, feature_weights, covariance_estimator=sandwich)
            L_S.fit()

            S = L_S.summary('twosided')
            P_S = [p for p, v in zip(S['pval'], S['variable']) if v not in active]

            return P_P, P_S

def test_logistic_pvals(n=500,
                        p=200,
                        s=3,
                        sigma=2,
                        rho=0.3,
                        snr=7.):

    counter = 0

    while True:
        counter += 1

        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr)

        z = (y > 0)
        X = np.hstack([np.ones((n,1)), X])

        active = np.array(active)
        active += 1
        active = [0] + list(active)

        L = lasso.logistic(X, z, [0]*1 + [1.2]*p)
        L.fit()
        S = L.summary('onesided')

        if set(active).issubset(L.active) > 0:
            return [p for p, v in zip(S['pval'], S['variable']) if v not in active]
        return []

def test_adding_quadratic_lasso():

    X, y, beta, active, sigma = instance(n=300, p=200)
    Q = identity_quadratic(0.01, 0, np.random.standard_normal(X.shape[1]), 0)

    L1 = lasso.gaussian(X, y, 20, quadratic=Q)
    beta1 = L1.fit(solve_args={'min_its':500, 'tol':1.e-12})
    G1 = X[:,L1.active].T.dot(X.dot(beta1) - y) + Q.objective(beta1,'grad')[L1.active]
    np.testing.assert_allclose(G1 * np.sign(beta1[L1.active]), -20)

    lin = identity_quadratic(0.0, 0, np.random.standard_normal(X.shape[1]), 0)
    L2 = lasso.gaussian(X, y, 20, quadratic=lin)
    beta2 = L2.fit(solve_args={'min_its':500, 'tol':1.e-12})
    G2 = X[:,L2.active].T.dot(X.dot(beta2) - y) + lin.objective(beta2,'grad')[L2.active]
    np.testing.assert_allclose(G2 * np.sign(beta2[L2.active]), -20)
