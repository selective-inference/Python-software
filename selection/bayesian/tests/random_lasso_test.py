import numpy as np
import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.randomized.api import randomization
from selection.bayesian.paired_bootstrap import pairs_bootstrap_glm, bootstrap_cov

n = 100
p = 20
s = 5
snr = 5

sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
X_1, y, true_beta, nonzero, noise_variance = sample.generate_response()
random_Z = np.random.standard_normal(p)
sel = selection(X_1, y, random_Z, randomization_scale=1, sigma=None, lam=None)
lam, epsilon, active, betaE, cube, initial_soln = sel
print true_beta, active

bootstrap_score = pairs_bootstrap_glm(rr.glm.gaussian(X_1,y), active, beta_full=None, inactive = ~active)[0]
sampler = lambda: np.random.choice(n, size=(n,),replace = True)
#print(sampler)
cov = bootstrap_cov(sampler, bootstrap_score)




