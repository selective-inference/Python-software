from __future__ import division
import numpy as np
import numpy.testing.decorators as dec

from selection.sqrt_lasso import (sqrt_lasso, choose_lambda,
                                  estimate_sigma, data_carving, split_model)
from selection.lasso import instance
from selection.affine import constraints_unknown_sigma
from selection.truncated import T as truncated_T

from selection.tests.test_sample_ball import _generate_constraints

def test_class(n=20, p=40, s=2):
    y = np.random.standard_normal(n) * 1.2
    beta = np.zeros(p)
    beta[:s] = 5
    X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
    y += np.dot(X, beta)
    lam_theor = choose_lambda(X, quantile=0.9)
    L = sqrt_lasso(y,X,lam_theor)
    L.fit(tol=1.e-10, min_its=80)
    P = []
    if L.active.shape[0] > 0:

        np.testing.assert_array_less( \
            np.dot(L.constraints.linear_part, L.y),
            L.constraints.offset)

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

@dec.slow
def test_goodness_of_fit(n=20, p=25, s=10, sigma=20.,
                         nsample=100):
    P = []
    while True:
        y = np.random.standard_normal(n) * sigma
        beta = np.zeros(p)
        X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
        X /= (X.std(0)[None,:] * np.sqrt(n))
        y += np.dot(X, beta) * sigma
        lam_theor = .7 * choose_lambda(X, quantile=0.9)
        L = sqrt_lasso(y, X, lam_theor)
        L.fit(tol=1.e-12, min_its=150)

        pval = L.goodness_of_fit(lambda x: np.max(np.fabs(x)),
                                 burnin=10000,
                                 ndraw=10000)
        P.append(pval)
        Pa = np.array(P)
        Pa = Pa[~np.isnan(Pa)]
        print (~np.isnan(np.array(Pa))).sum()
        if (~np.isnan(np.array(Pa))).sum() >= nsample:
            break
        print np.mean(Pa), np.std(Pa)

    return Pa

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

def test_pval_intervals(nsample=100):
    pvalues = []
    gaussian_pvalues = []
    coverage = 0
    count = 0
    for _ in range(nsample):
        P, P_gaussian, intervals, beta = test_gaussian_approx()

        if P != []:
            pvalues.extend(P)
            gaussian_pvalues.extend(P_gaussian)
            for i, C in intervals:
                count += 1
                if beta[i] <= C[1] and beta[i] >= C[0]:
                    coverage += 1

    return pvalues, gaussian_pvalues, coverage/count
            

def test_data_carving(n=100,
                      p=200,
                      s=7,
                      sigma=5,
                      rho=0.3,
                      snr=7.,
                      split_frac=0.9,
                      lam_frac=1.,
                      ndraw=8000,
                      burnin=2000):

    counter = 0

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr)
        L, stage_one = split_model(y, X, 
                        lam_frac=lam_frac,
                        split_frac=split_frac)[:2]

        print counter, L.active

        if set(range(s)).issubset(L.active):
            results, L = data_carving(y, X, lam_frac=lam_frac, 
                                      stage_one=stage_one,
                                      splitting=True, 
                                      ndraw=ndraw,
                                      burnin=burnin)

            carve = [r[1] for r in results]
            split = [r[3] for r in results]
            return carve[s:], split[s:], carve[:s], split[:s], counter



if __name__ == "__main__":
    #P, IS = main()
    pass
