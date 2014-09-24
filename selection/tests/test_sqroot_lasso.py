import numpy as np
from selection.sqrt_lasso import (sqrt_lasso, choose_lambda,
                                  estimate_sigma)
from selection.affine import constraints_unknown_sigma
from selection.truncated_T import truncated_T

def test_class(n=20, p=40, s=2):
    y = np.random.standard_normal(n) * 1.2
    beta = np.zeros(p)
    beta[:s] = 3
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
    return P #, I

def test_estimate_sigma(n=200, p=400, s=10, sigma=3.):
    y = np.random.standard_normal(n) * sigma
    beta = np.zeros(p)
    beta[:s] = 8 * (2 * np.random.binomial(1, 0.5, size=(s,)) - 1)
    X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    y += np.dot(X, beta) * sigma
    lam_theor = choose_lambda(X, quantile=0.9)
    L = sqrt_lasso(y, X, lam_theor)
    L.fit(tol=1.e-10, min_its=80)
    P = []

    return L.sigma_hat / sigma, L.sigma_E / sigma, L.df_E

#     if L.active.shape[0] > 0:

#         np.testing.assert_array_less( \
#             np.dot(L.constraints.linear_part, L.y),
#             L.constraints.offset)

#         if set(range(s)).issubset(L.active):
#             value = L.sigma_hat / sigma, L.sigma_E / sigma, L.df_E
#         else:
#             value = (None,)*3
#     else:
#         value = (None,)*3
#     return value

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
                                                       DEBUG=True)
            df = np.diag(R).sum()
            truncT = truncated_T(np.array([(interval.lower_value,
                                            interval.upper_value) for interval in intervals]), df)
            sigma_hat = np.linalg.norm(np.dot(R, Z)) / np.sqrt(df)
            print truncT.intervals, ((eta*Z).sum() / np.linalg.norm(eta)) / sigma_hat, obs, 'observed', intervals
            sf = truncT.sf(obs)
            pval = 2 * min(sf, 1.-sf)

            P.append(float(pval))
            IS.append(truncT.intervals)

    return P, IS
    

if __name__ == "__main__":
    #P, IS = main()
    pass
