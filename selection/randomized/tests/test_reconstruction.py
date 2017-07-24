from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.decorators import wait_for_return_value, register_report
import selection.tests.reports as reports

from selection.api import multiple_queries
from selection.randomized.glm import split_glm_group_lasso, target as glm_target
from selection.tests.instance import logistic_instance

@wait_for_return_value()
def test_reconstruction(s=3,
                        n=200,
                        p=50, 
                        signal=7,
                        rho=0.1,
                        split_frac=0.8,
                        lam_frac=0.7,
                        ndraw=100, 
                        burnin=200, 
                        bootstrap=True,
                        solve_args={'min_its':50, 'tol':1.e-10},
                        reference_known=False): 

    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)

    m = int(split_frac * n)
    nonzero = np.where(beta)[0]

    loss = rr.glm.logistic(X, y)
    epsilon = 1. / np.sqrt(n)

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 2000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = split_glm_group_lasso(loss, epsilon, m, penalty)
    mv = multiple_queries([M_est])
    mv.solve()

    M_est.selection_variable['variables'] = M_est.selection_variable['variables']
    nactive = np.sum(M_est.selection_variable['variables'])

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(M_est.selection_variable['variables'])[0]):

        active_set = np.nonzero(M_est.selection_variable['variables'])[0]

        target_sampler, target_observed = glm_target(loss, 
                                                     M_est.selection_variable['variables'],
                                                     mv)

        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin,
                                              keep_opt=True)
        
        reconstruction = target_sampler.reconstruction_map(target_sample)
        logdens = target_sampler.log_randomization_density(target_sample)
        return logdens.shape
