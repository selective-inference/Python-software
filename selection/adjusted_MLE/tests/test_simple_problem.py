from __future__ import print_function
import numpy as np, sys

from scipy.stats import norm as ndist
from selection.adjusted_MLE.selective_MLE import solve_UMVU
from selection.adjusted_MLE.tests.exact_MLE import grad_CGF, fisher_info
from statsmodels.distributions.empirical_distribution import ECDF
from selection.adjusted_MLE.tests.approx_MLE import approx_fisher_info

def simple_problem(target_observed=2, n=1, threshold=2, randomization_scale=1., epsilon = 0.05):
    """
    Simple problem: randomizaiton of sd 1 and thresholded at 2 (default args)
    """
    target_observed = np.atleast_1d(target_observed)
    target_transform = (-np.identity(n), np.zeros(n))
    opt_transform = (np.identity(n)+ epsilon, np.ones(n) * threshold)
    feasible_point = np.ones(n)
    randomizer_precision = np.identity(n) / randomization_scale ** 2
    target_cov = np.identity(n)

    return solve_UMVU(target_transform,
                      opt_transform,
                      target_observed,
                      feasible_point,
                      target_cov,
                      randomizer_precision)


def sim_simple_problem(true_mean, threshold=2, randomization_scale=1., epsilon = 0.05):
    while True:
        Z, W = np.random.standard_normal(2)
        Z += true_mean
        W *= randomization_scale
        if ((Z + W) - threshold)/(1.+epsilon)>0.:
            return Z


def check_unbiased(true_mean, threshold=2, randomization_scale=1., nsim=5000, epsilon = 0.05):
    bias = 0
    for _ in range(nsim):
        Z = sim_simple_problem(true_mean, threshold, randomization_scale)
        est = simple_problem(Z, threshold=threshold, randomization_scale=randomization_scale)[0]
        bias += est - true_mean

    return bias / nsim

#print(check_unbiased(-1., threshold=2, randomization_scale=1., nsim=5000, epsilon = 0.05))

def test_orthogonal_lasso(n=5):
    Zval = np.random.normal(0, 1, n)
    print("observed Z" + str(Zval) + "\n")
    approx_MLE = simple_problem(Zval, threshold=2, randomization_scale=1.)[0]

    approx_MLE2 = [simple_problem(z, threshold=2, randomization_scale=1.)[0] for z in Zval]
    mu_seq = np.linspace(-6, 6, 2500)
    grad_partition = np.array([grad_CGF(mu, randomization_scale=1., threshold=2) for mu in mu_seq])

    exact_MLE = []
    for k in range(Zval.shape[0]):
        mle = mu_seq[np.argmin(np.abs(grad_partition - Zval[k]))]
        exact_MLE.append(mle)

    return approx_MLE, np.asarray(exact_MLE), np.asarray(approx_MLE2)


def bootstrap_simple(n= 100, B=100, true_mean=0., threshold=2.):

    resid_matrix = np.identity(n) - np.ones((n,n)) / n
    U, D, V = np.linalg.svd(resid_matrix)
    U = U[:,:-1]

    while True:
        target_Z, omega = np.random.standard_normal(2)
        target_Z += true_mean * np.sqrt(n)
        if target_Z + omega > threshold:
            Zval = U.dot(np.random.standard_normal(n-1))
            Zval += target_Z * np.ones(n) / np.sqrt(n)
            break

    approx_MLE, value, mle_map = simple_problem(target_Z, n=1, threshold=2, randomization_scale=1.)

    boot_sample = []
    for b in range(B):
        Zval_boot = np.sum(Zval[np.random.choice(n, n, replace=True)]) / np.sqrt(n)
        boot_sample.append(mle_map(Zval_boot)[0])

    print("approx_MLE", approx_MLE, np.std(boot_sample), true_mean)
    return boot_sample, np.mean(boot_sample), np.std(boot_sample), \
           np.squeeze((boot_sample - np.mean(boot_sample)) / np.std(boot_sample)), \
           np.true_divide(approx_MLE - np.sqrt(n)*true_mean, np.std(boot_sample))

def check_approx_fisher_simple(true_mean, threshold=2, randomization_scale=1., nsim=200):
    diff = 0.
    for _ in range(nsim):
        Z = sim_simple_problem(true_mean, threshold, randomization_scale)
        approx = simple_problem(Z, threshold=threshold, randomization_scale=randomization_scale)
        approx_std = np.sqrt(np.diag(approx[2]))

        exact_std = 1./np.sqrt(fisher_info(approx[0], randomization_scale = 1., threshold = 2))
        diff += np.abs(exact_std-approx_std)
        print("difference", np.abs(exact_std-approx_std))

    print(diff/float(nsim))

def pivot_approx_fisher_simple(n=100, true_mean = 0., threshold=2, epsilon = 0.2):

    while True:
        target_Z, omega = np.random.standard_normal(2)
        target_Z += true_mean * np.sqrt(n)
        if ((target_Z + omega) - threshold)/(1.+epsilon)>0.:
            break

    n1 =1
    target_observed = np.atleast_1d(target_Z)
    target_transform = (-np.identity(n1), np.zeros(n1))
    #s = np.asscalar(np.sign(target_Z + omega))
    opt_transform = ((np.identity(n1)+epsilon), np.ones(n1) * (threshold))
    feasible_point = np.ones(n1)
    randomization_scale = 1.
    randomizer_precision = np.identity(n1) / randomization_scale ** 2
    target_cov = np.identity(n1)
    simple_var = 1./approx_fisher_info(target_observed, randomization_scale=1., threshold=2)

    approx_MLE, value, var, mle_map = solve_UMVU(target_transform,
                                                 opt_transform,
                                                 target_observed,
                                                 feasible_point,
                                                 target_cov,
                                                 randomizer_precision)

    print("approx MLE", approx_MLE, np.sqrt(n)*true_mean, var)
    print("diff", simple_var- var)
    return np.squeeze((approx_MLE - np.sqrt(n)*true_mean)/np.sqrt(var)), approx_MLE - np.sqrt(n)*true_mean, \
           np.squeeze((approx_MLE - np.sqrt(n)*true_mean)/np.sqrt(simple_var)), simple_var- var


#test_matrices_simple(true_mean=2., threshold=2, epsilon=0.2)

# if __name__ == "__main__":
#     n = 1000
#     Zval = np.random.normal(0, 1, n)
#     sys.stderr.write("observed Z" + str(Zval) + "\n")
#     MLE = simple_problem(Zval, n=n, threshold=2, randomization_scale=1.)[0]
#     #print(MLE)
#
#     mu_seq = np.linspace(-6, 6, 200)
#     grad_partition = np.array([grad_CGF(mu, randomization_scale=1., threshold=2) for mu in mu_seq])
#
#     exact_MLE = []
#     for k in range(Zval.shape[0]):
#         mle = mu_seq[np.argmin(np.abs(grad_partition - Zval[k]))]
#         exact_MLE.append(mle)
#
#     np.testing.assert_allclose(MLE, exact_MLE, rtol=2.0)

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     plt.clf()
#     Zval = np.linspace(-5, 5, 51)
#     MLE = np.array([simple_problem(z)[0] for z in Zval])
#
#     mu_seq = np.linspace(-6, 6, 200)
#     grad_partition = np.array([grad_CGF(mu, randomization_scale=1., threshold=2) for mu in mu_seq])
#
#     plt.plot(Zval, MLE, label='+2')
#     plt.plot(grad_partition, mu_seq, 'r--', label='MLE')
#     plt.legend()
#     plt.show()

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     ndraw = 200
#     boot_pivot=[]
#     for i in range(ndraw):
#         boot_result = bootstrap_simple(n=300, B=5000, true_mean=0., threshold=2.)
#         boot_pivot.append(boot_result[4])
#
#         print("boot sample", np.asarray(boot_pivot).shape, boot_pivot)
#         ecdf = ECDF(ndist.cdf(np.asarray(boot_pivot)))
#         grid = np.linspace(0, 1, 101)
#
#         if i % 10 == 0:
#             plt.clf()
#             print("ecdf", ecdf(grid))
#             plt.plot(grid, ecdf(grid), c='red', marker='^')
#             plt.plot([0,1],[0,1], 'k--')
#             plt.savefig('bootstrap_simple.png')

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ndraw = 500
    pivot_obs_info=[]
    bias = 0.
    diff = 0.
    for i in range(ndraw):
        result = pivot_approx_fisher_simple(n=300, true_mean = -0.1, threshold=2)
        pivot_obs_info.append(result[0])
        diff += result[3]
        bias += result[1]
        sys.stderr.write("bias" + str(bias / float(i)) + "\n")

    sys.stderr.write("overall_bias" + str(bias / float(ndraw)) + "\n")
    sys.stderr.write("difference between variances" + str(diff / float(ndraw)) + "\n")

    ecdf = ECDF(ndist.cdf(np.asarray(pivot_obs_info)))
    grid = np.linspace(0, 1, 101)

    plt.clf()
    plt.plot(grid, ecdf(grid), c='red', marker='^')
    plt.plot([0,1],[0,1], 'k--')
    plt.show()
#   #plt.savefig('/Users/snigdhapanigrahi/Desktop/signed_approx_info_simple_amp_neg1.png')