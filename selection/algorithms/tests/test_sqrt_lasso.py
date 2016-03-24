from __future__ import division

import numpy as np
import numpy.testing.decorators as dec
import nose.tools as nt

# make any plots not use display

from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt

# used for ECDF

import statsmodels.api as sm

from selection.algorithms.sqrt_lasso import (sqrt_lasso, choose_lambda,
                                  estimate_sigma, data_carving, split_model)
import selection.algorithms.sqrt_lasso as SQ
from selection.algorithms.lasso import instance
from selection.constraints.quasi_affine import constraints_unknown_sigma
from selection.truncated import T as truncated_T
from selection.sampling.tests.test_sample_sphere import _generate_constraints
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_for_test

def test_class(n=20, p=40, s=2):
    y = np.random.standard_normal(n) * 1.2
    beta = np.zeros(p)
    beta[:s] = 5
    X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
    y += np.dot(X, beta)
    lam_theor = 0.7 * choose_lambda(X, quantile=0.9)
    L = sqrt_lasso(y,X,lam_theor)
    L.fit(tol=1.e-10, min_its=80)
    P = []
    if L.active.shape[0] > 0:

        np.testing.assert_array_less( \
            np.dot(L.constraints.linear_part, L.y),
            L.constraints.offset)

        nt.assert_true(L.constraints(y))
        nt.assert_true(L.quasi_affine_constraints(y))

        if set(range(s)).issubset(L.active):
            P = [p[1] for p in L.active_pvalues[s:]]
        else:
            P = []
    return P

def test_estimate_sigma(n=200, p=400, s=10, sigma=3.):

    y = np.random.standard_normal(n) * sigma
    beta = np.zeros(p)
    beta[:s] = 8 * (2 * np.random.binomial(1, 0.5, size=(s,)) - 1)
    X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    y += np.dot(X, beta) * sigma
    lam_theor = choose_lambda(X, quantile=0.9)
    L = sqrt_lasso(y, X, lam_theor)
    L.fit(tol=1.e-12, min_its=150)
    P = []

    if L.active.shape[0] > 0:

        return L.sigma_hat / sigma, L.sigma_E / sigma, L.df_E
    else:
        return (None,) * 3

@set_sampling_params_iftrue(True)
@dec.slow
def test_goodness_of_fit(n=20, p=25, s=10, sigma=20.,
                         nsim=1000, burnin=2000, ndraw=8000):
    P = []
    while True:
        y = np.random.standard_normal(n) * sigma
        beta = np.zeros(p)
        X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
        X /= (X.std(0)[None,:] * np.sqrt(n))
        y += np.dot(X, beta) * sigma
        lam_theor = .7 * choose_lambda(X, quantile=0.9)
        L = sqrt_lasso(y, X, lam_theor)
        L.fit(tol=1.e-12, min_its=150, max_its=200)

        pval = L.goodness_of_fit(lambda x: np.max(np.fabs(x)),
                                 burnin=burnin,
                                 ndraw=ndraw)
        P.append(pval)
        Pa = np.array(P)
        Pa = Pa[~np.isnan(Pa)]
        if (~np.isnan(np.array(Pa))).sum() >= nsim:
            break

    U = np.linspace(0,1,101)
    plt.plot(U, sm.distributions.ECDF(Pa)(U))
    plt.plot([0,1], [0,1])
    plt.savefig("goodness_of_fit_uniform", format="pdf")


def test_class_R(n=100, p=20):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    lam_theor = choose_lambda(X, quantile=0.25)
    L = sqrt_lasso(y,X,lam_theor)
    L.fit(tol=1.e-7)

    if L.active.shape[0] > 0:
        np.testing.assert_array_less( \
            np.dot(L.constraints.linear_part, L.y),
            L.constraints.offset)

        return L.active_constraints.linear_part, L.active_constraints.offset / L.sigma_E, L.R_E, L._XEinv[0]
    else:
        return None, None, None, None

def test_gaussian_approx(n=100,p=200,s=10):
    """
    using gaussian approximation for pvalues
    """
    sigma = 3
    y = np.random.standard_normal(n) * sigma
    beta = np.zeros(p)
    #beta[:s] = 8 * (2 * np.random.binomial(1, 0.5, size=(s,)) - 1)
    beta[:s] = 18 
    X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    y += np.dot(X, beta)
    lam_theor = choose_lambda(X, quantile=0.75)
    L = sqrt_lasso(y, X, lam_theor)
    L.fit(tol=1.e-10, min_its=80)

    P = []
    P_gaussian = []
    intervals = []
    if L.active.shape[0] > 0:

        np.testing.assert_array_less( \
            np.dot(L.constraints.linear_part, L.y),
            L.constraints.offset)

        if set(range(s)).issubset(L.active):
            P = [p[1] for p in L.active_pvalues[s:]]
            P_gaussian = [p[1] for p in L.active_gaussian_pval[s:]]
            intervals = [u for u in L.active_gaussian_intervals if u[0] in range(s)]
    return P, P_gaussian, intervals, beta


@set_sampling_params_iftrue(True)
def test_pval_intervals(nsim=100, burnin=None, ndraw=None):
    pvalues = []
    gaussian_pvalues = []
    coverage = 0
    count = 0
    for _ in range(nsim):
        P, P_gaussian, intervals, beta = test_gaussian_approx()

        if P != []:
            pvalues.extend(P)
            gaussian_pvalues.extend(P_gaussian)
            for i, L, U in intervals:
                count += 1
                if beta[i] <= U and beta[i] >= L:
                    coverage += 1

    return pvalues, gaussian_pvalues, coverage/count
            


@set_sampling_params_iftrue(True)
def test_data_carving(n=100,
                      p=200,
                      s=7,
                      rho=0.3,
                      snr=7.,
                      split_frac=0.8,
                      lam_frac=1.,
                      ndraw=8000,
                      burnin=2000, 
                      df=np.inf,
                      coverage=0.90,
                      sigma=3,
                      fit_args={'min_its':120, 'tol':1.e-12},
                      compute_intervals=True,
                      nsim=None):

    counter = 0

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
        L, stage_one = split_model(y, 
                                   X, 
                                   lam_frac=lam_frac,
                                   split_frac=split_frac,
                                   fit_args=fit_args)[:2]

        print L.active
        if set(range(s)).issubset(L.active):
            results, L = data_carving(y, X, lam_frac=lam_frac, 
                                      stage_one=stage_one,
                                      splitting=True, 
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      coverage=coverage,
                                      fit_args=fit_args,
                                      compute_intervals=compute_intervals)

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

            return carve[s:], split[s:], carve[:s], split[:s], counter, carve_coverage, split_coverage

def main_sigma(nsample=1000, sigma=3, s=10):
    S = []
    for _ in range(nsample):
        try:
            v = test_estimate_sigma(sigma=sigma, s=s)
            if v[0] is not None:
                S.append((v[0],v[1]))
        except (IndexError, ValueError):
            print 'exception raised'
            
        print np.mean(S, 0), np.std(S, 0)

def main(nsample=1000):

    while True:
        A, b, R, eta = test_class_R(n=10,p=6)
        if A is not None:
            break

    def sample(A, b, R, eta):
        n = A.shape[1]
        df = np.diag(R).sum()
        counter = 0
        while True:
            counter += 1
            Z = np.random.standard_normal(n) * 1.5
            sigma_hat = np.linalg.norm(np.dot(R, Z)) / np.sqrt(df)
            if np.all(np.dot(A, Z) <= b * sigma_hat):
                return Z
            if counter >= 1000:
                break
        return None

    P = []
    IS = []
    for i in range(nsample):
        Z = sample(A, b, R, eta)
        if Z is not None:
            print 'new sample'
            intervals, obs = constraints_unknown_sigma(A, b, Z, eta, R,
                                                       value_under_null=0.,
                                                       DEBUG=False)
            df = np.diag(R).sum()
            truncT = truncated_T(np.array([(interval.lower_value,
                                            interval.upper_value) for interval in intervals]), df)
            sigma_hat = np.linalg.norm(np.dot(R, Z)) / np.sqrt(df)
            #print truncT.intervals, ((eta*Z).sum() / np.linalg.norm(eta)) / sigma_hat, obs, 'observed', intervals
            sf = truncT.sf(obs)
            pval = 2 * min(sf, 1.-sf)

            P.append(float(pval))
            IS.append(truncT.intervals)

    return P#, IS
    
@set_seed_for_test()
def test_skinny_fat():

    X, Y = instance()[:2]
    n, p = X.shape
    lam = SQ.choose_lambda(X)
    obj1 = SQ.sqlasso_objective(X, Y)
    obj2 = SQ.sqlasso_objective_skinny(X, Y)
    soln1 = SQ.solve_sqrt_lasso_fat(X, Y, min_its=500, weights=np.ones(p) * lam)
    soln2 = SQ.solve_sqrt_lasso_skinny(X, Y, min_its=500, weights=np.ones(p) * lam)

    np.testing.assert_almost_equal(soln1, soln2)

    X, Y = instance(p=50)[:2]
    n, p = X.shape
    lam = SQ.choose_lambda(X)
    obj1 = SQ.sqlasso_objective(X, Y)
    obj2 = SQ.sqlasso_objective_skinny(X, Y)
    soln1 = SQ.solve_sqrt_lasso_fat(X, Y, min_its=500, weights=np.ones(p) * lam)
    soln2 = SQ.solve_sqrt_lasso_skinny(X, Y, min_its=500, weights=np.ones(p) * lam)

    np.testing.assert_almost_equal(soln1, soln2)

