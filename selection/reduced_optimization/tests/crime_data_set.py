import os, numpy as np, pandas, statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso


NRTI =  pandas.read_table("/Users/snigdhapanigrahi/Results_bayesian/trimmed_communities.txt", sep='\s+')
print("shape of data", NRTI.shape)
print("data types", NRTI.dtypes)

NRTI = NRTI.as_matrix()
#print("first row of data", NRTI[0,:])

for i in range(97):
X_NRTI = np.array(NRTI[:,:97])
#X_NRTI = X_NRTI.astype(float)
Y = NRTI[:,97] # shorthand
Y = Y.astype(float)
Y = np.array(np.log(Y), np.float); Y -= Y.mean()
X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
X = X_NRTI # shorthand
n, p = X.shape
X /= np.sqrt(n)