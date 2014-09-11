import numpy as np
import selection.affine; reload(selection.affine)
from selection.affine import constraints_unknown_sigma
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

def simulate(theta=0):

    n = 22
    p = 4
    k = 18
    R = np.linalg.svd(np.random.standard_normal((n,n-k)), full_matrices=0)[0]
    R = np.dot(R, R.T)
    R = 0.1 * R + np.diag([0]*p + [1.] * (n-p))
    R = np.linalg.svd(R, full_matrices=0)[0]
    R = R[:,:(n-p)]
    R = np.dot(R, R.T)
    A = np.diag([1.]*p) + 0.05 * np.random.standard_normal((p,p))
    sel = np.identity(n)[:p]
    A = np.dot(A, sel)
    b = -np.ones(p)
    n = R.shape[0]
    df = np.diag(R).sum()

    eta = np.random.standard_normal(n) * 3
    eta = eta - np.dot(R, eta)

    counter = 0
    while True:
        counter += 1
        Z = np.random.standard_normal(n) * 1.5 + eta * theta / np.linalg.norm(eta)**2
        sigma_hat = np.linalg.norm(np.dot(R, Z)) / np.sqrt(df)
        if np.all(np.dot(A, Z) <= b * sigma_hat):
            break
    return A, b, R, Z, eta, counter


def instance(theta=0):

    A, b, R, Z, eta, counter = simulate(theta=theta)
    print counter
    from selection.truncated_T import truncated_T
    
    intervals, obs = constraints_unknown_sigma(A, b, Z, eta, R,
                                               value_under_null=theta)
    df = np.diag(R).sum()
    truncT = truncated_T(np.array([(i.lower_value,
                                    i.upper_value) for i in intervals]), df)
    sf = truncT.sf(obs)
    pval = 2 * min(sf, 1.-sf)
    if pval < 1.e-6:
        print sf, obs, intervals
    return float(pval)

if __name__ == "__main__":
    
    P = []
    for i in range(2000):
        P.append(instance(theta=1.))
        print i, np.mean(P), np.std(P)
    U = np.linspace(0,1,51)
    plt.plot(U, ECDF(P)(U))
    plt.plot([0,1],[0,1])
    plt.show()
