from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np

#arguments of function are X with normalized columns, response y, sigma_hat and randomization
def simes_selection(X, y, sigma_hat, alpha, randomizer= 'gaussian'):

    n, p = X.shape

    if randomizer == 'gaussian':
        perturb = np.random.standard_normal(p)

    T_stats = X.T.dot(y)/sigma_hat
    randomized_T_stats = np.true_divide(T_stats, sigma_hat) + perturb
    randomized_T_stats_order = np.sort(np.abs(randomized_T_stats))
    index_sort = np.argsort(np.abs(randomized_T_stats))
    order_index = np.arange(p)

    cut_offs = 1.- (np.arange(p)+1.)*(alpha/(2.*p))
    threshold = normal.ppf(cut_offs,0.,1.)

    if np.any(randomized_T_stats_order>= threshold):

        exceed_T =  order_index[randomized_T_stats_order>= threshold]

        t_0 = exceed_T[0]

        i_0 = index_sort[t_0]

        J = index_sort[:t_0]

        T_stats_inactive = T_stats[J]

        T_stats_active = T_stats[i_0]

        #print(exceed_T, t_0, index_sort, i_0, J)

        return T_stats_inactive, T_stats_active, i_0, np.sign(T_stats_active)

    else:

        return 0.,0.,0.,0.











