from __future__ import print_function
import time
from scipy.stats import norm as normal
import os, numpy as np, pandas, statsmodels.api as sm
from selection.randomized.api import randomization
from selection.tests.instance import gaussian_instance
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection

n = 100
p = 50
s = 0
snr = 0.

gene_0 = pandas.read_table("/Users/snigdhapanigrahi/0.txt", na_values="NA")
print("shape of data",gene_0.shape[0], gene_0.shape[1])
X = np.array(gene_0.ix[:,2:])
print("shape of X", X.shape)
y=  np.array(gene_0.ix[:,1])
print("shape of y",y.shape)

#print("norms of columns of X", np.linalg.norm(X[:,2]))

#X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr, random_signs=True)
print(simes_selection(X, y, sigma_hat= 1. , alpha=0.05, randomizer= 'gaussian'))

