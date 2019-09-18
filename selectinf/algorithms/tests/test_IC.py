from copy import copy

import numpy as np
from ...tests.instance import gaussian_instance
from ...constraints.affine import sample_from_constraints
from ...distributions.discrete_family import discrete_family

from ..forward_step import info_crit_stop

def test_data_carving_IC(n=600,
                         p=100,
                         s=10,
                         sigma=5,
                         rho=0.25,
                         signal=(3.5,5.),
                         split_frac=0.9,
                         ndraw=25000,
                         burnin=5000, 
                         df=np.inf,
                         coverage=0.90,
                         compute_intervals=False):

    X, y, beta, active, sigma, _ = gaussian_instance(n=n, 
                                                     p=p, 
                                                     s=s, 
                                                     sigma=sigma, 
                                                     rho=rho, 
                                                     signal=signal, 
                                                     df=df,
                                                     equicorrelated=False)
    mu = np.dot(X, beta)
    splitn = int(n*split_frac)
    indices = np.arange(n)
    np.random.shuffle(indices)
    stage_one = indices[:splitn]

    FS = info_crit_stop(y, X, sigma, cost=np.log(n), subset=stage_one)

    con = FS.constraints()

    X_E = X[:,FS.active]
    X_Ei = np.linalg.pinv(X_E)
    beta_bar = X_Ei.dot(y)
    mu_E = X_E.dot(beta_bar)
    sigma_E = np.linalg.norm(y-mu_E) / np.sqrt(n - len(FS.active))

    con.mean[:] = mu_E
    con.covariance = sigma_E**2 * np.identity(n)

    print(sigma_E, sigma)
    Z = sample_from_constraints(con, 
                                y,
                                ndraw=ndraw,
                                burnin=burnin)
    
    pvalues = []
    for idx, var in enumerate(FS.active):
        active = copy(FS.active)
        active.remove(var)
        X_r = X[:,active] # restricted design
        mu_r = X_r.dot(np.linalg.pinv(X_r).dot(y))
        delta_mu = (mu_r - mu_E) / sigma_E**2

        W = np.exp(Z.dot(delta_mu))
        fam = discrete_family(Z.dot(X_Ei[idx].T), W)
        pval = fam.cdf(0, x=beta_bar[idx])
        pval = 2 * min(pval, 1 - pval)
        pvalues.append((pval, beta[var]))

    return pvalues
