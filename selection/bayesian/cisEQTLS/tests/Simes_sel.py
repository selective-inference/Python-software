from __future__ import print_function
import time
from scipy.stats import norm as normal
import os, numpy as np, pandas, statsmodels.api as sm
from selection.randomized.api import randomization
from selection.tests.instance import gaussian_instance
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.bayesian.cisEQTLS.randomized_lasso_sel import selection

gene_0 = pandas.read_table("/Users/snigdhapanigrahi/tiny_example/2.txt", na_values="NA")
print("shape of data", gene_0.shape[0], gene_0.shape[1])
X = np.array(gene_0.ix[:,2:])
n, p = X.shape
#X = np.sqrt(n)* X

y = np.sqrt(n) * np.array(gene_0.ix[:,1])

print(simes_selection(X, y, alpha=0.05, randomizer= 'gaussian'))
#random_Z = np.random.standard_normal(p)
#sel = selection(X, y, random_Z, sigma= 1.)
#lam, epsilon, active, betaE, cube, initial_soln = sel
#print("selection via Lasso", betaE, lam, active.sum(), [i for i in range(p) if active[i]])

