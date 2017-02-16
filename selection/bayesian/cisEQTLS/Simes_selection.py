from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np

#this function takes the nominal p-values as input, computes the Simes test statistic to delclare if gene is significant or not
def simes_pvalue(p_value):

    p = p_value.shape[0]
    p_sorted = np.sort(p_value)

    p_simes = (p/(np.arange(p) + 1.))* p_sorted

    return p_simes

#arguments of function are X with normalized columns, response y, sigma_hat and randomization
def simes_selection(X, y, alpha, randomizer= 'gaussian'):

    n, p = X.shape

    if randomizer == 'gaussian':
        perturb = np.random.standard_normal(p)

    sigma_hat = np.sqrt((1.- (X.T.dot(y)**2))/(n-2))
    T_stats = X.T.dot(y)/sigma_hat
    tau = np.max(sigma_hat)/5.
    perturb = tau * perturb

    randomized_T_stats = T_stats + perturb
    randomized_T_stats_order = np.sort(np.abs(randomized_T_stats))
    p_val_randomized = np.sort(2*(1. - normal.cdf(np.abs(randomized_T_stats))))

    indices_order = np.argsort(2*(1. - normal.cdf(np.abs(randomized_T_stats))))
    indices = np.arange(p)

    simes_p_randomized = np.min((p/(np.arange(p) + 1.))* p_val_randomized)

    if simes_p_randomized <= alpha/2:

        significant =  indices_order[p_val_randomized <= alpha/2]

        i_0 = significant[0]

        t_0 = indices[p_val_randomized <= ((np.arange(p) + 1.)/(2*p))* alpha]

        J = indices_order[:t_0[0]]

        T_stats_inactive = T_stats[J]

        T_stats_active = T_stats[i_0]

        T_stats_inf = np.append(T_stats_inactive,T_stats_active)

        return p_val_randomized-((np.arange(p) + 1.)/(2*p))* alpha, simes_p_randomized, i_0, np.sign(T_stats_active), T_stats_inf

    else:

        return simes_p_randomized


















