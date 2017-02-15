from __future__ import print_function
import time
from scipy.stats import norm as normal
import numpy as np
from selection.randomized.api import randomization
from selection.tests.instance import gaussian_instance
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection

n = 100
p = 50
s = 0
snr = 0.

X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr, random_signs=True)
simes_selection(X, y, sigma_hat= 1. , alpha=0.05, randomizer= 'gaussian')

