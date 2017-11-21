from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import solve_UMVU
from statsmodels.distributions.empirical_distribution import ECDF

def BH_selection(p_values, level):

    m = p_values.shape[0]
    p_sorted = np.sort(p_values)
    indices = np.arange(m)
    indices_order = np.argsort(p_values)
    order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
    E_sel = indices_order[:(order_sig+1)]

    active = np.zeros(m, np.bool)
    active[E_sel] = 1
    return order_sig+1, active


def orthogonal_lasso_approx(n=100, s=3, signal=3, randomization_scale=1., sigma = 1., level=0.10):

    while True:
        beta = np.zeros(n)

        signal = np.atleast_1d(signal)
        if signal.shape == (1,):
            beta[:s] = signal[0] * (1 + np.fabs(np.random.standard_normal(s)))
        else:
            beta[:s] = np.linspace(signal[0], signal[1], s)

        y = sigma * (beta + np.random.standard_normal(n))
        omega = randomization_scale * np.random.standard_normal(n)

        p_values = 2.*(1. - ndist.cdf(np.abs(y+omega)/np.sqrt(1.+ randomization_scale**2.)))
        K, active = BH_selection(p_values, level)

        threshold = np.sqrt(1.+ randomization_scale**2.)*ndist.ppf(1.-(K*level)/n)
        target_observed = y[active]
        target_transform = (-np.identity(K), np.zeros(K))
        s = np.sign(target_observed + omega[active])
        opt_transform = (np.identity(K)*s[None, :], threshold*s*np.ones(K))
        nactive = np.sum(active)
        feasible_point= np.ones(nactive)

        if nactive >0:
            true_target = beta[active]
            print("true_target", true_target)
            approx_MLE, value, var, mle_map = solve_UMVU(target_transform,
                                                         opt_transform,
                                                         target_observed,
                                                         feasible_point,
                                                         sigma*np.identity(nactive),
                                                         randomization_scale*np.identity(nactive))

            print("approx sd", np.sqrt(np.diag(var)))
            break

    return np.true_divide((approx_MLE - true_target),np.sqrt(np.diag(var))), (approx_MLE - true_target).sum() / float(nactive)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ndraw = 500
    bias = 0.
    pivot_obs_info= []
    for i in range(ndraw):
        approx = orthogonal_lasso_approx(n=100, s=20, signal=2.5, randomization_scale=1., sigma = 1., level=0.10)
        if approx is not None:
            pivot = approx[0]
            bias += approx[1]
            print("bias in iteration", approx[1])
            pivot_obs_info.extend(pivot)

        sys.stderr.write("iteration completed" + str(i) + "\n")
        sys.stderr.write("overall_bias" + str(bias / float(i)) + "\n")

    plt.clf()
    ecdf = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf(grid), c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()