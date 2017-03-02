from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np

#this function passes p-values through a BH-sieve to declare the significant ones at pre-fixed level
def BH_q(p_value, level):

    m = p_value.shape[0]
    p_sorted = np.sort(p_value)
    indices = np.arange(m)
    indices_order = np.argsort(p_value)

    #print("sorted p values", p_sorted-np.true_divide(level*(np.arange(m)+1.),2.*m))
    if np.any(p_sorted - np.true_divide(level*(np.arange(m)+1.),m)<=np.zeros(m)):
        order_sig = np.max(indices[p_sorted- np.true_divide(level*(np.arange(m)+1.),m)<=0])
        sig_pvalues = indices_order[:order_sig]
        return p_sorted[:order_sig], sig_pvalues

    else:
        return None


# this function takes the nominal p-values as input, computes the Simes test statistic to delclare if gene is significant or not

def simes_pvalue(p_value):

    p = p_value.shape[0]
    p_sorted = np.sort(p_value)

    p_simes = (p/(np.arange(p) + 1.))* p_sorted

    return p_simes

#arguments of function are X with normalized columns, response y, sigma_hat and randomization
def simes_selection(X, y, alpha, randomizer= 'gaussian', randomization_scale = 1.):

    n, p = X.shape

    if randomizer == 'gaussian':
        perturb = np.random.standard_normal(p)

    sigma_hat = 1.
    T_stats = X.T.dot(y)/sigma_hat

    randomized_T_stats = T_stats + randomization_scale * perturb
    p_val_randomized = np.sort(2*(1. - normal.cdf(np.abs(randomized_T_stats))))

    indices_order = np.argsort(2*(1. - normal.cdf(np.abs(randomized_T_stats))))
    indices = np.arange(p)

    simes_p_randomized = np.min((p/(np.arange(p) + 1.))* p_val_randomized)

    #print("simes_p", simes_p_randomized)

    if simes_p_randomized <= alpha:

        significant =  indices_order[p_val_randomized <= alpha]

        i_0 = significant[0]

        T_stats_active = T_stats[i_0]

        t_0 = indices[p_val_randomized <= ((np.arange(p) + 1.)/(p))* alpha]

        if t_0[0] > 0:
            J = indices_order[:t_0[0]]

        else:
            J = -1*np.ones(1)

        return i_0, J, t_0[0], np.sign(T_stats_active)

    else:

        return None


















