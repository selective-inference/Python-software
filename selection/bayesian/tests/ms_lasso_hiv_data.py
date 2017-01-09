import os, numpy as np, pandas, statsmodels.api as sm
import time
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
import matplotlib.pyplot as plt
from selection.bayesian.ms_lasso_2stage import selection_probability_objective_ms_lasso, sel_prob_gradient_map_ms_lasso,\
    selective_map_credible_ms_lasso


if not os.path.exists("NRTI_DATA.txt"):
    NRTI = pandas.read_table("http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt", na_values="NA")
else:
    NRTI = pandas.read_table("NRTI_DATA.txt")

NRTI_specific = []
NRTI_muts = []
mixtures = np.zeros(NRTI.shape[0])
for i in range(1,241):
    d = NRTI['P%d' % i]
    for mut in np.unique(d):
        if mut not in ['-','.'] and len(mut) == 1:
            test = np.equal(d, mut)
            if test.sum() > 10:
                NRTI_specific.append(np.array(np.equal(d, mut)))
                NRTI_muts.append("P%d%s" % (i,mut))

NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)
X_NRTI = np.array(NRTI_specific, np.float)
Y = NRTI['3TC'] # shorthand
keep = ~np.isnan(Y).astype(np.bool)
X_NRTI = X_NRTI[np.nonzero(keep)]; Y=Y[keep]
Y = np.array(np.log(Y), np.float); Y -= Y.mean()
X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
X = X_NRTI # shorthand
n, p = X.shape
X /= np.sqrt(n)

ols_fit = sm.OLS(Y, X).fit()
sigma_3TC = np.linalg.norm(ols_fit.resid) / np.sqrt(n-p-1)
OLS_3TC = ols_fit.params

print("noise_varaince", sigma_3TC)
random_Z = np.random.standard_normal(p)
Z_stats = X.T.dot(Y)
randomized_Z_stats = np.true_divide(Z_stats, sigma_3TC) + random_Z

active_1 = np.zeros(p, bool)
active_1[np.fabs(randomized_Z_stats) > 1.96] = 1
active_signs_1 = np.sign(randomized_Z_stats[active_1])
nactive_1 = active_1.sum()
print("active_1",active_1, nactive_1)

threshold = 1.96 * np.ones(p)
X_step2 = X[:, ~active_1]
random_Z_2 = np.random.standard_normal(p - nactive_1)
sel = selection(X_step2, Y, random_Z_2)
lam, epsilon, active_2, betaE, cube, initial_soln = sel
noise_variance = sigma_3TC ** 2
lagrange = lam * np.ones(p-nactive_1)
nactive_2 = betaE.shape[0]
print("active_2", active_2, nactive_2)
active_signs_2 = np.sign(betaE)

primal_feasible_1 = np.fabs(randomized_Z_stats[active_1])
primal_feasible_2 = np.fabs(betaE)
feasible_point = np.append(primal_feasible_1, primal_feasible_2)

active = np.zeros(p, bool)
active[active_1] = 1
indices_stage2 = np.where(active == 0)[0]
active[indices_stage2[active_2]] = 1
nactive = active.sum()

print("active", active, nactive)

randomizer = randomization.isotropic_gaussian((p,), 1.)
parameter = np.random.standard_normal(nactive)
mean = X[:, active].dot(parameter)
generative_X = X[:, active]
print("shape of generative X", generative_X.shape)
prior_variance = 100.

Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(Y))
post_var = (prior_variance* np.identity(nactive)) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])
unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])
print(unadjusted_intervals)

grad_map = sel_prob_gradient_map_ms_lasso(X,
                                          feasible_point,  # in R^{|E|_1 + |E|_2}
                                          active_1,  # the active set chosen by randomized marginal screening
                                          active_2,  # the active set chosen by randomized lasso
                                          active_signs_1,  # the set of signs of active coordinates chosen by ms
                                          active_signs_2,  # the set of signs of active coordinates chosen by lasso
                                          lagrange,  # in R^p
                                          threshold,  # in R^p
                                          generative_X,  # in R^{p}\times R^{n}
                                          noise_variance,
                                          randomizer,
                                          epsilon)

ms = selective_map_credible_ms_lasso(Y,
                                     grad_map,
                                     prior_variance)

sel_MAP = ms.map_solve(nstep=500)[::-1]

#print("selective MAP- ms_lasso", sel_MAP[1])

toc = time.time()
samples = ms.posterior_samples()
tic = time.time()
print('sampling time', tic - toc)

adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
adjusted_intervals = np.vstack([sel_MAP[1], adjusted_intervals])
print("selective map and intervals", adjusted_intervals)
