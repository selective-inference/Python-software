from __future__ import print_function
import numpy as np
from selection.tests.instance import gaussian_instance
import regreg.api as rr

from selection.randomized.M_estimator import restricted_Mest
from selection.randomized.M_estimator_nonrandom import M_estimator

def test_nonrandomized(s=0,
                       n=200,
                       p=10,
                       snr=7,
                       rho=0,
                       lam_frac=0.8,
                       solve_args={'min_its': 20, 'tol': 1.e-10}):

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
    nonzero = np.where(beta)[0]
    loss = rr.glm.gaussian(X, y)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = M_estimator(loss, penalty)
    M_est.solve()
    active  = M_est._overall
    nactive = np.sum(active)
    print("nactive",nactive)
    if nactive == 0:
        return None
    beta_unpenalized = restricted_Mest(loss, active, solve_args=solve_args)

    M_est.setup_sampler(np.zeros(p), n)
    #M_est.sample(ndraw = 1000, burnin=1000, stepsize=1./p)

    test_stat = lambda x: np.linalg.norm(x[:nactive])

    return M_est.hypothesis_test(test_stat, test_stat(M_est.observed_score_state), stepsize=1./p)


if __name__=='__main__':

    pvals = []
    for i in range(50):
        print(i)
        pval = test_nonrandomized()
        print(pval)
        if pval is not None:
            pvals.append(pval)

    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    fig = plt.figure()
    ax = fig.gca()

    ecdf = sm.distributions.ECDF(pvals)
    G = np.linspace(0, 1)
    F = ecdf(G)
    ax.plot(G, F, '-o', c='b', lw=2, label='CLT')
    ax.plot([0, 1], [0, 1], 'k-', lw=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()