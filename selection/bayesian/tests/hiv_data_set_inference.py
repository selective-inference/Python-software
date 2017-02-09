import os, numpy as np, pandas, statsmodels.api as sm
import time
import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.inference_rr import sel_prob_gradient_map, selective_map_credible

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
print("here")

# Next, standardize the data, keeping only those where Y is not missing

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

# Design matrix
# Columns are site / amino acid pairs
print(X.shape)

#solving the Lasso at theoretical lambda
tau = 1.0
print(tau**2)
random_Z = np.random.normal(loc=0.0, scale= tau, size= p)
sel = selection(X, Y, random_Z, sigma=sigma_3TC)

lam, epsilon, active, betaE, cube, initial_soln = sel

print("value of tuning parameter",lam)
print("nactive", active.sum())

active_set_0 = [NRTI_muts[i] for i in range(p) if active[i]]
print("active variables", active_set_0)
active_set = [i for i in range(p) if active[i]]

noise_variance = sigma_3TC**2
nactive = betaE.shape[0]
active_signs = np.sign(betaE)

primal_feasible = np.fabs(betaE)
dual_feasible = np.ones(p)
dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
lagrange = lam * np.ones(p)
generative_X = X[:, active]
prior_variance = 1000.

Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(Y))
post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])
unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])
#print(unadjusted_intervals)

inf_rr = selective_map_credible(Y,
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

map = inf_rr.map_solve_2(nstep = 500)[::-1]

toc = time.time()
samples = inf_rr.posterior_samples()
tic = time.time()
print('sampling time', tic - toc)

adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
sel_mean = np.mean(samples, axis=0)
print("active variables", active_set_0)
print("selective mean", sel_mean)
print("selective map", map[1])
print("selective map and intervals", adjusted_intervals)
print("usual posterior based map & intervals", unadjusted_intervals)
