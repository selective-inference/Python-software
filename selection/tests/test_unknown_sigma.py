import numpy as np
import selection.affine; reload(selection.affine)
from selection.affine import constraints_unknown_sigma
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

def simulate_orth():

    n = 22
    p = 4
    R = residual_projector = np.diag([0]*p+[1.]*(n-p))
    A = np.diag([1.]*p) + 0.05 * np.random.standard_normal((p,p))
    sel = np.identity(n)[:p]
    A = np.dot(A, sel)
    b = -np.ones(p)
    n = R.shape[0]
    df = np.diag(R).sum()

    while True:
        Z = np.random.standard_normal(n) * 1.5
        sigma_hat = np.linalg.norm(np.dot(R, Z)) / np.sqrt(df)
        if np.all(np.dot(A, Z) <= b * sigma_hat):
            break
    eta = np.random.standard_normal(n)
    eta[p:] = 0
    return A, b, R, Z, eta

def simulate():

    n = 22
    p = 4
    R = residual_projector = np.diag([0]*(p-2)+[1.]*(n-p+2))
    A = np.diag([1.]*p) + 0.05 * np.random.standard_normal((p,p))
    sel = np.identity(n)[:p]
    A = np.dot(A, sel)
    b = -np.ones(p)
    n = R.shape[0]
    df = np.diag(R).sum()

    while True:
        Z = np.random.standard_normal(n) * 1.5
        sigma_hat = np.linalg.norm(np.dot(R, Z)) / np.sqrt(df)
        if np.all(np.dot(A, Z) <= b * sigma_hat):
            break
    eta = np.random.standard_normal(n)
    eta[(p-2):] = 0.
    return A, b, R, Z, eta


def instance():

    A, b, R, Z, eta = simulate_orth()

    from selection.truncated_T import truncated_T
    
    intervals, obs = constraints_unknown_sigma(A, b, Z, eta, R)
    df = np.diag(R).sum()
    truncT = truncated_T(np.array([(i.lower_value,
                                    i.upper_value) for i in intervals]), df)
    sf = truncT.sf(obs)
    pval = 2 * min(sf, 1.-sf)
    if pval < 1.e-6:
        print sf, obs, intervals
    return float(pval)

if __name__ == "__main__":
    
    P = [instance() for _ in range(500)]
    plt.plot(U, ECDF(P)(U))
