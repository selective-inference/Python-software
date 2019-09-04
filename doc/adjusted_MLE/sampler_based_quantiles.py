import numpy as np
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.tests.instance import gaussian_instance
from selection.constraints.affine import constraints, sample_from_constraints
from scipy.stats import norm
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats

def compute_sampler_quantiles(n=500, 
                              p=100, 
                              signal_fac=1.2, 
                              s=5, 
                              sigma=1., 
                              rho=0., 
                              randomizer_scale=1, 
                              full_dispersion=True):

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        if full_dispersion:
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        true_mean = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        estimate, observed_info_mean, _, pval, intervals, _ = conv.selective_MLE(observed_target,
                                                                                 cov_target,
                                                                                 cov_target_score,
                                                                                 alternatives)

        opt_linear, opt_offset = conv.opt_transform
        target_precision = np.linalg.inv(cov_target)
        randomizer_cov, randomizer_precision = conv.randomizer.cov_prec
        score_linear = np.identity(p)
        target_linear = score_linear.dot(cov_target_score.T.dot(target_precision))
        target_offset = conv.observed_score_state - target_linear.dot(observed_target)

        nopt = opt_linear.shape[1]
        ntarget = target_linear.shape[1]

        implied_precision = np.zeros((ntarget + nopt, ntarget + nopt))
        implied_precision[:ntarget, :ntarget] = target_linear.T.dot(randomizer_precision).dot(target_linear) + target_precision
        implied_precision[:ntarget, ntarget:] = target_linear.T.dot(randomizer_precision).dot(opt_linear)
        implied_precision[ntarget:, :ntarget] = opt_linear.T.dot(randomizer_precision).dot(target_linear)
        implied_precision[ntarget:, ntarget:] = opt_linear.T.dot(randomizer_precision).dot(opt_linear)
        implied_cov = np.linalg.inv(implied_precision)

        conditioned_value = target_offset + opt_offset
        implied_mean = implied_cov.dot(np.hstack((target_precision.dot(true_mean)-target_linear.T.dot(randomizer_precision).dot(conditioned_value),
                                                  -opt_linear.T.dot(randomizer_precision).dot(conditioned_value))))

        A_scaling = np.zeros((nopt, ntarget+nopt))
        A_scaling[:,ntarget:] = -np.identity(nopt)
        b_scaling = np.zeros(nopt)
        affine_con = constraints(A_scaling,
                                 b_scaling,
                                 mean=implied_mean,
                                 covariance=implied_cov)

        initial_point = np.zeros(ntarget+nopt)
        initial_point[ntarget:] = conv.observed_opt_state

        sampler = sample_from_constraints(affine_con,
                                          initial_point,
                                          ndraw=500000,
                                          burnin=1000)

        print("sampler", sampler.shape, sampler[:,:ntarget].shape)
        mle_sample = []
        for j in range(sampler.shape[0]):
            estimate, _, _, _, _, _ = conv.selective_MLE(sampler[j,:ntarget],
                                                         cov_target,
                                                         cov_target_score,
                                                         alternatives)
            mle_sample.append(estimate)
            print("iteration ", j)
        mle_sample = np.asarray(mle_sample)
        print("check", mle_sample.shape, np.mean(mle_sample, axis=0) - true_mean)

        for i in range(nonzero.sum()):
            temp = 251 + i
            ax = plt.subplot(temp)
            stats.probplot(mle_sample[:,i], dist="norm", plot=pylab)
            plt.subplots_adjust(hspace=.5, wspace=.5)
        pylab.show()

        sampler_quantiles = np.vstack([np.percentile(mle_sample, 5, axis=0), np.percentile(mle_sample, 95, axis=0)])

        normal_quantiles = np.vstack((norm.ppf(0.05, loc=true_mean, scale=np.sqrt(np.diag(observed_info_mean))),
                                      norm.ppf(0.95, loc=true_mean, scale=np.sqrt(np.diag(observed_info_mean)))))

        print("sampler quantiles", sampler_quantiles.T)
        print("normal quantiles", normal_quantiles.T)
        break


