from __future__ import print_function
import numpy as np
import selection.constraints.affine
from selection.constraints.quasi_affine import constraints_unknown_sigma

def simulate(A=None, theta=0, R=None, eta=None):

    n = 22
    p = 4
    k = 18
    if R is None:
        R = np.linalg.svd(np.random.standard_normal((n,n-k)), full_matrices=0)[0]
        R = np.dot(R, R.T)
        R = 0.1 * R + np.diag([0]*p + [1.] * (n-p))
        R = np.linalg.svd(R, full_matrices=0)[0]
        R = R[:,:(n-p)]
        R = np.dot(R, R.T)
    if A is None:
        A = np.diag([1.]*p) + 0.05 * np.random.standard_normal((p,p))
        sel = np.identity(n)[:p]
        A = np.dot(A, sel)
    b = -np.ones(p)
    n = R.shape[0]
    df = np.diag(R).sum()

    if eta is None:
        eta = np.random.standard_normal(n) * 3
        eta = eta - np.dot(R, eta)

    counter = 0
    while True:
        counter += 1
        Z = np.random.standard_normal(n) * 1.5 + eta * theta / np.linalg.norm(eta)**2
        sigma_hat = np.linalg.norm(np.dot(R, Z)) / np.sqrt(df)
        if np.all(np.dot(A, Z) <= b * sigma_hat):
            return A, b, R, Z, eta, counter
        if counter >= 1000:
            break
    return None


def instance(theta=0, A=None, R=None, eta=None):

    result = None
    while not result:
        result = simulate(theta=theta, A=A, R=R, eta=eta)

    A, b, R, Z, eta, counter = result
    from selection.truncated_T import truncated_T
    
    intervals, obs = constraints_unknown_sigma(A, b, Z, eta, R,
                                               value_under_null=theta)
    df = np.diag(R).sum()
    truncT = truncated_T(np.array([(interval.lower_value,
                                    interval.upper_value) for interval in intervals]), df)
    sf = truncT.sf(obs)
    pval = 2 * min(sf, 1.-sf)
    if pval < 1.e-6:
        print(sf, obs, intervals)
    return float(pval)

if __name__ == "__main__":
    
    P = []

    n = 22
    p = 4
    k = 18

    A = np.diag([1.]*p) + 0.05 * np.random.standard_normal((p,p))
    sel = np.identity(n)[:p]
    A = np.dot(A, sel)

    R = np.linalg.svd(np.random.standard_normal((n,n-k)), full_matrices=0)[0]
    R = np.dot(R, R.T)
    R = 0.1 * R + np.diag([0]*p + [1.] * (n-p))
    R = np.linalg.svd(R, full_matrices=0)[0]
    R = R[:,:(n-p)]
    R = np.dot(R, R.T)

    eta = np.random.standard_normal(n) * 3
    eta = eta - np.dot(R, eta)

    for i in range(1000):
        P.append(instance(theta=3.,R=R, A=A, eta=eta))
        print(i, np.mean(P), np.std(P))
    U = np.linspace(0,1,51)

    # make any plots not use display

    from matplotlib import use
    use('Agg')
    import matplotlib.pyplot as plt

    # used for ECDF

    import statsmodels.api as sm
    plt.plot(U, sm.distributions.ECDF(P)(U))
    plt.plot([0,1],[0,1])
    plt.show()
