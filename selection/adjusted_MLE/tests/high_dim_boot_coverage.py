from __future__ import print_function
from rpy2 import robjects

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import statsmodels.api as sm
import numpy as np, sys
import regreg.api as rr
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU
import scipy.stats as stats

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    library(bestsubset)
    sim_xy = bestsubset::sim.xy
    ''')

    r_simulate = robjects.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

def inference_approx(n=100, p=1000, nval=100, rho=0.35, s=5, beta_type=2, snr=0.2,
                     randomization_scale=np.sqrt(0.25), target="partial"):
    while True:
        X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        true_mean = X.dot(beta)

        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))

        X_val -= X_val.mean(0)[None, :]
        X_val /= (X_val.std(0)[None, :] * np.sqrt(nval))

        if p > n:
            sigma_est = np.std(y)
            print("sigma and sigma_est", sigma, sigma_est)
        else:
            ols_fit = sm.OLS(y, X).fit()
            sigma_est = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1.)
            print("sigma and sigma_est", sigma, sigma_est)

        y = y - y.mean()
        y /= sigma_est
        y_val = y_val - y_val.mean()
        y_val /= sigma_est
        true_mean /= sigma_est

        loss = rr.glm.gaussian(X, y)
        epsilon = 1. / np.sqrt(n)
        lam_seq = np.linspace(0.75, 2.75, num=100) * np.mean(
            np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        err = np.zeros(100)
        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        M = np.identity(p)
        for k in range(100):
            lam = lam_seq[k]
            W = np.ones(p) * lam
            penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
            M_est = M_estimator_map(loss, epsilon, penalty, randomizer, M, target=target,
                                    randomization_scale=randomization_scale, sigma=1.)

            active = M_est._overall
            nactive = active.sum()
            approx_MLE_est = np.zeros(p)
            if nactive > 0:
                M_est.solve_map()
                approx_MLE = solve_UMVU(M_est.target_transform,
                                        M_est.opt_transform,
                                        M_est.target_observed,
                                        M_est.feasible_point,
                                        M_est.target_cov,
                                        M_est.randomizer_precision)[0]
                approx_MLE_est[active] = approx_MLE

            err[k] = np.mean((y_val - X_val.dot(approx_MLE_est)) ** 2.)

        lam = lam_seq[np.argmin(err)]
        print('lambda', lam)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, M, target=target,
                                randomization_scale=randomization_scale, sigma=1.)
        active = M_est._overall
        nactive = np.sum(active)

        print("number of variables selected by randomized LASSO", nactive)

        if nactive > 0:
            M_est.solve_map()
            approx_MLE, var, mle_map, _, _, mle_transform = solve_UMVU(M_est.target_transform,
                                                                       M_est.opt_transform,
                                                                       M_est.target_observed,
                                                                       M_est.feasible_point,
                                                                       M_est.target_cov,
                                                                       M_est.randomizer_precision)

            approx_sd = np.sqrt(np.diag(var))

            if nactive == 1:
                approx_MLE = np.array([approx_MLE])
                approx_sd = np.array([approx_sd])

            coverage_sel = 0.
            if target == "full":
                true_target = np.linalg.pinv(X)[active].dot(true_mean)
            if target == "partial":
                true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(true_mean)
            print("true target", true_target)

            for j in range(nactive):
                if (approx_MLE[j] - (1.65 * approx_sd[j])) <= true_target[j] and (approx_MLE[j] + (1.65 * approx_sd[j])) >= true_target[j]:
                    coverage_sel += 1

                print("selective intervals", sigma_est*(approx_MLE[j] - (1.65 * approx_sd[j])),
                      sigma_est *(approx_MLE[j] + (1.65 * approx_sd[j])))

            break

    if True:
        return coverage_sel/float(nactive), np.true_divide(approx_MLE- true_target, approx_sd)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    ndraw = 100
    coverage_sel = 0.
    pivot_obs_info = []
    for i in range(ndraw):
        approx = inference_approx(n=500, p=2500, nval=500, rho=0.35, s=20, beta_type=1, snr=0.20, target="full")
        if approx is not None:
            coverage_sel += approx[0]
            pivot = approx[1]
            for j in range(pivot.shape[0]):
                pivot_obs_info.append(pivot[j])

        sys.stderr.write("selective coverage" + str(coverage_sel / float(i + 1)) + "\n")
        sys.stderr.write("iteration completed" + str(i) + "\n")
        #sys.stderr.write("pivot" + str(pivot_obs_info) + "\n")

    stats.probplot(np.asarray(pivot_obs_info), dist="norm", plot=plt)
    plt.show()
    #plt.savefig("/Users/snigdhapanigrahi/Desktop/high_10_0.20_.png")



