from __future__ import print_function
import time
from scipy.stats import norm as normal
import os, numpy as np, pandas, statsmodels.api as sm
from selection.randomized.api import randomization
from selection.tests.instance import gaussian_instance
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection, BH_q
from selection.bayesian.cisEQTLS.initial_sol_wocv import selection, instance
from selection.bayesian.cisEQTLS.randomized_lasso_sel import selection

def Simes_sel_test():
    gene_0 = pandas.read_table("/Users/snigdhapanigrahi/tiny_example/2.txt", na_values="NA")
    print("shape of data", gene_0.shape[0], gene_0.shape[1])
    X = np.array(gene_0.ix[:,2:])
    n, p = X.shape()
    y = np.sqrt(n) * np.array(gene_0.ix[:,1])
    print(simes_selection(X, y, alpha=0.05, randomizer= 'gaussian'))
#random_Z = np.random.standard_normal(p)
#sel = selection(X, y, random_Z, sigma= 1.)
#lam, epsilon, active, betaE, cube, initial_soln = sel
#print("selection via Lasso", betaE, lam, active.sum(), [i for i in range(p) if active[i]])

def BH_test():
    m = 100
    p_values = np.append(0.00002*np.ones(10), np.random.uniform(low=0, high=1.0, size=m))
    p_BH = BH_q(p_values, 0.05)

    if p_BH is not None:
        print("results from BH", p_BH[0], p_BH[1])

#BH_test()

def Simes_sel_test_0(n):

    X, y, true_beta, nonzero, noise_variance = sample.generate_response()

    sel_simes = simes_selection(X, y, alpha=0.10/40000, randomizer='gaussian')

    if sel_simes is None:

        return 0

    else:

        return 1


n = 350
p = 5000
s = 0
snr = 5.

sample = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

nsel = 0

for i in range(20000):

    nsel += Simes_sel_test_0(n = 20000)

print(nsel)

