from __future__ import print_function
import numpy as np, sys

import regreg.api as rr
from scipy.stats import norm as ndist
from selection.randomized.api import randomization
from selection.tests.instance import gaussian_instance
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
    return order_sig+1, active, p_values[indices_order[order_sig+1]]

def orthogonal_BH_approx(n=100, s=3, signal=3, randomization_scale=1., sigma = 1., level=0.10):

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
        K, active, p_threshold = BH_selection(p_values, level)

        threshold = np.sqrt(1.+ randomization_scale**2.)*ndist.ppf(1.-np.max((K*level)/n, p_threshold))
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


def BH_approx(n=100, p=50, s=5, signal=5., randomization_scale=1., sigma=1., level=0.10):

    while True:

        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=0.2, signal=signal, sigma=sigma,
                                                       random_signs=False, equicorrelated=False)

        omega = randomization_scale * np.random.standard_normal(p)
        p_values = 2.*(1. - ndist.cdf(np.abs(X.T.dot(y)+omega)/np.sqrt(1.+ randomization_scale**2.)))
        K, active, p_threshold = BH_selection(p_values, level)
        nactive = active.sum()

        if nactive >0:

            threshold = np.sqrt(1. + randomization_scale ** 2.) * ndist.ppf(1.-max((K*level)/n, p_threshold))

            X_active_inv = np.linalg.inv(X[:, active].T.dot(X[:, active]))
            projection_perp = np.identity(n) - X[:, active].dot(X_active_inv).dot(X[:, active].T)
            observed_score_state = np.hstack(
                [np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(y),
                 X[:, ~active].T.dot(projection_perp).dot(y)])
            target_observed = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(y)
            true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
            active_signs = np.sign(X[:, active].T.dot(y) + omega[active])

            _opt_linear_term = np.vstack([np.diag(active_signs), np.zeros((p - nactive,nactive))])
            _opt_affine_term = np.concatenate([threshold * active_signs, X[:, ~active].T.dot(y) + omega[~active]])
            opt_transform = (_opt_linear_term, _opt_affine_term)

            _score_linear_term = np.zeros((p, p))
            _score_linear_term[:nactive, :nactive] = -X[:, active].T.dot(X[:, active])
            _score_linear_term[nactive:, :nactive] = -X[:, ~active].T.dot(X[:, active])
            _score_linear_term[nactive:, nactive:] = -np.identity(p - nactive)

            score_cov = np.zeros((p, p))
            score_cov[:nactive, :nactive] = X_active_inv
            score_cov[nactive:, nactive:] = X[:, ~active].T.dot(projection_perp).dot(X[:, ~active])
            score_target_cov = score_cov[:, :nactive]
            target_cov = score_cov[:nactive, :nactive]

            A = np.dot(_score_linear_term, score_target_cov).dot(np.linalg.inv(target_cov))
            data_offset = _score_linear_term.dot(observed_score_state) - A.dot(target_observed)
            target_transform = (A, data_offset)

            feasible_point = np.ones(nactive)

            approx_MLE, value, var, mle_map = solve_UMVU(target_transform,
                                                         opt_transform,
                                                         target_observed,
                                                         feasible_point,
                                                         sigma*np.identity(nactive),
                                                         randomization_scale*np.identity(p))

            #print("approx sd", np.sqrt(np.diag(var)))
            break

    return np.true_divide((approx_MLE - true_target),np.sqrt(np.diag(var))), (approx_MLE - true_target).sum() / float(nactive)

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     ndraw = 500
#     bias = 0.
#     pivot_obs_info= []
#     for i in range(ndraw):
#         approx = orthogonal_BH_approx(n=100, s=20, signal=2.5, randomization_scale=1., sigma = 1., level=0.10)
#         if approx is not None:
#             pivot = approx[0]
#             bias += approx[1]
#             print("bias in iteration", approx[1])
#             pivot_obs_info.extend(pivot)
#
#         sys.stderr.write("iteration completed" + str(i) + "\n")
#         sys.stderr.write("overall_bias" + str(bias / float(i)) + "\n")
#
#     plt.clf()
#     ecdf = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
#     grid = np.linspace(0, 1, 101)
#     plt.plot(grid, ecdf(grid), c='red', marker='^')
#     plt.plot(grid, grid, 'k--')
#     plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ndraw = 500
    bias = 0.
    pivot_obs_info= []
    for i in range(ndraw):
        approx = BH_approx(n=1000, p=2000, s=100, signal=3.5, randomization_scale=1., sigma=1., level=0.10)
        if approx is not None:
            pivot = approx[0]
            bias += approx[1]
            print("bias in iteration", approx[1])
            pivot_obs_info.extend(pivot)

        sys.stderr.write("iteration completed" + str(i) + "\n")
        sys.stderr.write("overall_bias" + str(bias / float(i+1)) + "\n")

    plt.clf()
    ecdf = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf(grid), c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()