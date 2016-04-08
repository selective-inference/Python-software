import numpy as np
import numpy.testing.decorators as dec

from itertools import product
from selection.algorithms.lasso import (lasso, 
                                        data_carving, 
                                        instance, 
                                        split_model, 
                                        standard_lasso,
                                        instance, 
                                        nominal_intervals)
from regreg.api import identity_quadratic

from selection.tests.decorators import set_sampling_params_iftrue

def test_gaussian(n=100, p=20):

    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))

    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))
    Q = identity_quadratic(0.01, 0, np.ones(p), 0)

    weights_with_zeros = 0.1 * np.ones(p)
    weights_with_zeros[:3] = 0.

    for q, fw in product([Q, None],
                         [0.5*lam_theor, weights_with_zeros]):

        L = lasso.gaussian(X, y, fw, 1., quadratic=Q)
        L.fit()
        C = L.constraints

        I = L.intervals
        S = L.summary('onesided')
        S = L.summary('twosided')

        yield (np.testing.assert_array_less,
               np.dot(L.constraints.linear_part, L._onestep),
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
        C = L.constraints

        np.testing.assert_array_less( \
            np.dot(L.constraints.linear_part, L._onestep),
            L.constraints.offset)

        I = L.intervals
        P = L.summary()['pval']

        return L, C, I, P

def test_poisson():

    X = np.random.standard_normal((10,5))
    Y = np.random.poisson(10, size=(10,))

    L = lasso.poisson(X, Y, 0.1)
    L.fit()
    C = L.constraints

    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L._onestep),
        L.constraints.offset)

    I = L.intervals
    P = L.summary()['pval']

    return L, C, I, P

def test_coxph():

    Q = identity_quadratic(0.01, 0, np.ones(5), 0)
    X = np.random.standard_normal((100,5))
    T = np.random.standard_exponential(100)
    S = np.random.binomial(1, 0.5, size=(100,))

    L = lasso.coxph(X, T, S, 0.1, quadratic=Q)
    L.fit()
    C = L.constraints

    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L._onestep),
        L.constraints.offset)

    I = L.intervals
    P = L.summary()['pval']

    return L, C, I, P


@set_sampling_params_iftrue(True)
def test_data_carving(n=100,
                      p=200,
                      s=7,
                      sigma=5,
                      rho=0.3,
                      snr=7.,
                      split_frac=0.9,
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
        L, stage_one = split_model(y, X, 
                                   sigma=sigma,
                                   lam_frac=lam_frac,
                                   split_frac=split_frac)[:2]

        if set(range(s)).issubset(L.active):
            while True:
                results, L = data_carving(X, y, lam_frac=lam_frac, 
                                          sigma=sigma,
                                          stage_one=stage_one,
                                          splitting=True, 
                                          ndraw=ndraw,
                                          burnin=burnin,
                                          coverage=coverage,
                                          compute_intervals=compute_intervals)
                if set(range(s)).issubset(L.active):
                    print "succeed"
                    break
                print "failed at least once"

            carve = [r[1] for r in results]
            split = [r[3] for r in results]

            Xa = X[:,L.active]
            truth = np.dot(np.linalg.pinv(Xa), mu) 

            split_coverage = []
            carve_coverage = []
            for result, t in zip(results, truth):
                _, _, ci, _, si = result
                carve_coverage.append((ci[0] < t) * (t < ci[1]))
                split_coverage.append((si[0] < t) * (t < si[1]))

            TP = s
            FP = L.active.shape[0] - TP
            v = (carve[s:], split[s:], carve[:s], split[:s], counter, carve_coverage, split_coverage, TP, FP)
            return_value.append(v)
            break
        else:
            TP = len(set(L.active).intersection(range(s)))
            FP = L.active.shape[0] - TP
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
    las.summary()
    intervals = las.intervals
    nominal_intervals(las)
    t.append([(beta[I], L, U) for I, L, U in intervals])
    return t
    
def test_gaussian_pvals(n=100,
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
        L = lasso.gaussian(X, y, 20., sigma=sigma)
        L.fit()
        v = {1:'twosided',
             0:'onesided'}[counter % 2]
        if set(active).issubset(L.active):
            S = L.summary(v)
            return [p for p, v in zip(S['pval'], S['variable']) if v not in active]
        
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
