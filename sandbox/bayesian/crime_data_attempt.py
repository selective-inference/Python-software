
import os, numpy as np, pandas, statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import regreg.api as rr
from selection.reduced_optimization.initial_soln import selection
from selection.randomized.api import randomization
from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso

crime = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data', header=None, na_values=['?'])
crime = crime.iloc[:, 5:]
crime.dropna(inplace=True)
crime.head()

# define X and y
X = crime.iloc[:, :-1]
n, p = X.shape
X -= X.mean(0)[None, :]
X /= (X.std(0)[None, :] * np.sqrt(n))

Y = crime.iloc[:, -1]
print("shape", X.shape, Y.shape)

ols_fit = sm.OLS(Y, X).fit()
print("residual", np.linalg.norm(ols_fit.resid))
sigma_3TC = np.linalg.norm(ols_fit.resid) / np.sqrt(n-p-1)
OLS_3TC = ols_fit.params
print("sigma", sigma_3TC)
