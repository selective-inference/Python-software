from __future__ import division, print_function

import numpy as np
from scipy.stats import norm as ndist, invgamma
from scipy.linalg import fractional_matrix_power

from ..algorithms.barrier_affine import solve_barrier_affine_py


class posterior(object):
    """
    Parameters
    ----------
    observed_target : ndarray
        Observed estimate of target.
    cov_target : ndarray
        Estimated covariance of target.
    cov_target_score : ndarray
        Estimated covariance of target and score of randomized query.
    prior : callable
        A callable object that takes a single argument
        `parameter` of the same shape as `observed_target`
        and returns (value of log prior, gradient of log prior)
    dispersion : float, optional
        A dispersion parameter for likelihood.
    solve_args : dict
        Arguments passed to solver of affine barrier problem.
    """

    def __init__(self,
                 query,
                 observed_target,
                 cov_target,
                 regress_target_score,
                 dispersion,
                 prior,
                 solve_args={'tol': 1.e-12}):

        self.solve_args = solve_args

        linear_part = query.sampler.affine_con.linear_part
        offset = query.sampler.affine_con.offset

        opt_linear = query.opt_linear

        observed_score = query.observed_score_state + query.observed_subgrad

        print("dispersion ", dispersion)
        result, self.inverse_info, log_ref = query.selective_MLE(observed_target,
                                                                 cov_target,
                                                                 regress_target_score,
                                                                 dispersion)

        ### Note for an informative prior we might want to change this...

        self.cond_precision = np.linalg.inv(query.cond_cov)
        self.cov_target = cov_target
        self.prec_target = np.linalg.inv(cov_target)

        self.ntarget = self.cov_target.shape[0]
        self.nopt = self.cond_precision.shape[0]

        self.observed_target = observed_target
        self.regress_target_score = regress_target_score
        self.opt_linear = opt_linear
        self.observed_score = observed_score

        prod_score_prec = query.prod_score_prec_unnorm * dispersion
        M1 = prod_score_prec
        M2 = M1.dot(query.cov_rand).dot(M1.T)
        M3 = M1.dot(self.opt_linear.dot(query.cond_cov).dot(self.opt_linear.T)).dot(M1.T)

        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

        self.feasible_point = query.initial_point
        self.cond_mean = query.cond_mean
        self.linear_part = linear_part
        self.offset = offset

        self.initial_estimate = np.asarray(result['MLE'])
        self.dispersion = dispersion
        self.log_ref = log_ref

        self._set_marginal_parameters()

        self.prior = prior

    def log_posterior(self,
                      target_parameter,
                      sigma=1):

        """
        Parameters
        ----------
        target_parameter : ndarray
            Value of parameter at which to evaluate
            posterior and its gradient.
        sigma : ndarray
            Noise standard deviation.
        """

        sigmasq = sigma ** 2

        target = self.S.dot(target_parameter) + self.r

        mean_marginal = self.linear_coef.dot(target) + self.offset_coef
        prec_marginal = self.prec_marginal
        conjugate_marginal = prec_marginal.dot(mean_marginal)

        solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_marginal,
                                 prec_marginal,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **self.solve_args)

        log_normalizer = -val - mean_marginal.T.dot(prec_marginal).dot(mean_marginal) / 2.

        _prec = self.prec_target_nosel # shorthand
        log_lik = -((self.observed_target - target).T.dot(_prec).dot(self.observed_target - target)) / 2. \
                  - log_normalizer

        grad_lik = self.S.T.dot(_prec.dot(self.observed_target) - _prec.dot(target) - self.linear_coef.T.dot(
            prec_marginal.dot(soln) - conjugate_marginal))

        log_prior, grad_prior = self.prior(target_parameter)

        return (self.dispersion * (log_lik - self.log_ref) / sigmasq + log_prior,
                self.dispersion * grad_lik / sigmasq + grad_prior)

    ### Private method

    def _set_marginal_parameters(self):
        """
        This works out the implied covariance
        of optimization varibles as a function
        of randomization as well how to compute
        implied mean as a function of the true parameters.
        """

        T1 = self.regress_target_score.T.dot(self.prec_target)
        T2 = T1.T.dot(self.M2.dot(T1))
        T3 = T1.T.dot(self.M3.dot(T1))

        prec_target_nosel = self.prec_target + T2 - T3
        _P = -(T1.T.dot(self.M1.dot(self.observed_score)) + T2.dot(self.observed_target))

        _Q = np.linalg.inv(prec_target_nosel + T3)

        T4 = self.M1.T.dot(T1)
        T5 = self.opt_linear.T.dot(T4)
        T6 = self.cond_cov.dot(T5)
        T7 = self.opt_linear.dot(T6)
        T8 = self.M1.dot(T7)
        T9 = (-T8.dot(self.observed_target) + self.M1.dot(self.opt_linear.dot(self.cond_mean)))  ##flipped sign of first term here
        T10 = T1.T.dot(T9)

        self.prec_marginal = self.cond_precision - T5.dot(_Q).dot(T5)

        r = np.linalg.inv(prec_target_nosel).dot(T10 - _P)
        S = np.linalg.inv(prec_target_nosel).dot(self.prec_target)

        self.r = r
        self.S = S
        #print("check parameters for selected+lasso ", np.allclose(np.diag(S), np.ones(S.shape[0])), np.allclose(r, np.zeros(r.shape[0])))
        self.prec_target_nosel = prec_target_nosel


### sampling methods

def langevin_sampler(selective_posterior,
                     nsample=2000,
                     nburnin=100,
                     proposal_scale=None,
                     step=1.):
    state = selective_posterior.initial_estimate
    stepsize = 1. / (step * selective_posterior.ntarget)

    if proposal_scale is None:
        proposal_scale = selective_posterior.inverse_info

    sampler = langevin(state,
                       selective_posterior.log_posterior,
                       proposal_scale,
                       stepsize,
                       np.sqrt(selective_posterior.dispersion))

    samples = np.zeros((nsample, selective_posterior.ntarget))

    for i, sample in enumerate(sampler):
        sampler.scaling = np.sqrt(selective_posterior.dispersion)
        samples[i, :] = sample.copy()
        #print("sample ", i, samples[i,:])
        if i == nsample - 1:
            break

    return samples[nburnin:, :]


def gibbs_sampler(selective_posterior,
                  nsample=2000,
                  nburnin=100,
                  proposal_scale=None,
                  step=1.):
    state = selective_posterior.initial_estimate
    stepsize = 1. / (step * selective_posterior.ntarget)

    if proposal_scale is None:
        proposal_scale = selective_posterior.inverse_info

    sampler = langevin(state,
                       selective_posterior.log_posterior,
                       proposal_scale,
                       stepsize,
                       np.sqrt(selective_posterior.dispersion))
    samples = np.zeros((nsample, selective_posterior.ntarget))
    scale_samples = np.zeros(nsample)
    scale_update = np.sqrt(selective_posterior.dispersion)
    for i in range(nsample):
        sample = sampler.__next__()
        samples[i, :] = sample

        scale_update_sq = invgamma.rvs(a=(0.1 +
                                          selective_posterior.ntarget +
                                          selective_posterior.ntarget / 2),
                                       scale=0.1 - ((scale_update ** 2) * sampler.posterior_[0]),
                                       size=1)
        scale_samples[i] = np.sqrt(scale_update_sq)
        sampler.scaling = np.sqrt(scale_update_sq)

    return samples[nburnin:, :], scale_samples[nburnin:]


class langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 proposal_scale,
                 stepsize,
                 scaling):

        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           stepsize)
        self.proposal_scale = proposal_scale
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0, scale=1)
        self.sample = np.copy(initial_condition)
        self.scaling = scaling

        self.proposal_sqrt = fractional_matrix_power(self.proposal_scale, 0.5)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        while True:
            self.posterior_ = self.gradient_map(self.state, self.scaling)
            candidate = (self.state + self.stepsize * self.proposal_scale.dot(self.posterior_[1])
                         + np.sqrt(2.) * (self.proposal_sqrt.dot(self._noise.rvs(self._shape))) * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate, self.scaling)[1])):
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                break
        return self.state
