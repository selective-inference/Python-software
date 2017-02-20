from __future__ import print_function
import time
import os, numpy as np, pandas, statsmodels.api as sm
import numpy as np
from selection.tests.instance import gaussian_instance
#from selection.bayesian.cisEQTLS.randomized_lasso_sel import selection
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.lasso_reduced import sel_prob_gradient_map, selective_map_credible


gene_0 = pandas.read_table("/Users/snigdhapanigrahi/tiny_example/0.txt", na_values="NA")
print("shape of data", gene_0.shape[0], gene_0.shape[1])

X = np.array(gene_0.ix[:,2:])
n, p = X.shape

y = np.sqrt(n) * np.array(gene_0.ix[:,1])

random_Z = np.random.standard_normal(p)
sel = selection(X, y, random_Z, sigma= 1.)
lam, epsilon, active, betaE, cube, initial_soln = sel


nactive = active.sum()
active_set = [i for i in range(p) if active[i]]
print("active set", nactive, active_set)
print("active set of variables", lam, betaE)
active_signs = np.sign(betaE)
primal_feasible = np.fabs(betaE)
dual_feasible = np.ones(p)
dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
lagrange = lam * np.ones(p)
generative_X = X[:, active]
noise_variance = 1.
tau = 1.
prior_variance = noise_variance*100.


Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
print("OLS",post_mean)
post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])


inf_rr = selective_map_credible(y,
                                X,
                                primal_feasible,
                                dual_feasible,
                                active,
                                active_signs,
                                lagrange,
                                generative_X,
                                noise_variance,
                                prior_variance,
                                randomization.isotropic_gaussian((p,), tau),
                                epsilon)

toc = time.time()
samples = inf_rr.posterior_samples()
tic = time.time()
print('sampling time', tic - toc)

adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

print("unadjusted intervals", unadjusted_intervals)
print("adjusted intervals", adjusted_intervals)