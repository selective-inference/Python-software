from __future__ import division, print_function
import numpy as np, sys

from selectinf.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from scipy.stats import norm as ndist, invgamma
from scipy.linalg import fractional_matrix_power

class posterior_inference_lasso():

    def __init__(self,
                 observed_target,
                 cov_target,
                 cov_target_score,
                 feasible_point,
                 cond_mean,
                 cond_cov,
                 logdens_linear,
                 linear_part,
                 offset,
                 initial_estimate,
                 log_ref,
                 dispersion,
                 prior_var):

        self.ntarget = cov_target.shape[0]
        self.nopt = cond_cov.shape[0]

        self.cond_precision = np.linalg.inv(cond_cov)
        self.prec_target = np.linalg.inv(cov_target)

        self.observed_target = observed_target
        self.cov_target_score = cov_target_score
        self.logdens_linear = logdens_linear

        self.feasible_point = feasible_point
        self.cond_mean = cond_mean
        self.linear_part = linear_part
        self.offset = offset

        self.initial_estimate = initial_estimate
        self.prior_var = prior_var
        self.dispersion = dispersion
        self.log_ref = log_ref

        self.set_marginal_parameters()

    def set_marginal_parameters(self):

        target_linear = -self.logdens_linear.dot(self.cov_target_score.T.dot(self.prec_target))

        implied_precision = np.zeros((self.ntarget + self.nopt, self.ntarget + self.nopt))
        implied_precision[:self.ntarget, :self.ntarget] = (self.prec_target + target_linear.T.dot(self.cond_precision.dot(target_linear)))
        implied_precision[:self.ntarget, self.ntarget:] = -target_linear.T.dot(self.cond_precision)
        implied_precision[self.ntarget:, :self.ntarget] = (-target_linear.T.dot(self.cond_precision)).T
        implied_precision[self.ntarget:, self.ntarget:] = self.cond_precision

        implied_cov = np.linalg.inv(implied_precision)
        self.linear_coef = implied_cov[self.ntarget:, :self.ntarget].dot(self.prec_target)

        target_offset = self.cond_mean - target_linear.dot(self.observed_target)
        M = implied_cov[self.ntarget:, self.ntarget:].dot(self.cond_precision.dot(target_offset))
        N = -target_linear.T.dot(self.cond_precision).dot(target_offset)
        self.offset_coef = implied_cov[self.ntarget:, :self.ntarget].dot(N) + M

        self.cov_marginal = implied_cov[self.ntarget:, self.ntarget:]

    def prior(self, target_parameter, scale=1.):

        grad_prior = -target_parameter/(scale* self.prior_var)
        log_prior = -np.linalg.norm(target_parameter)**2 /(2.* scale * self.prior_var)

        return grad_prior, log_prior

    def log_posterior(self, target_parameter, scale=1., solve_args={'tol':1.e-12}):

        mean_marginal = self.linear_coef.dot(target_parameter) + self.offset_coef
        prec_marginal = np.linalg.inv(self.cov_marginal)
        conjugate_marginal = prec_marginal.dot(mean_marginal)

        solver = solve_barrier_affine_C

        val, soln, hess = solver(conjugate_marginal,
                                 prec_marginal,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        log_normalizer = -val - mean_marginal.T.dot(prec_marginal).dot(mean_marginal)/2.

        log_lik = -((self.observed_target - target_parameter).T.dot(self.prec_target).dot(self.observed_target - target_parameter)) / 2.\
                  - log_normalizer

        grad_lik = self.prec_target.dot(self.observed_target) - self.prec_target.dot(target_parameter) \
                   -self.linear_coef.T.dot(prec_marginal.dot(soln)- conjugate_marginal)

        grad_prior, log_prior = self.prior(target_parameter)

        return self.dispersion * grad_lik/scale + grad_prior, self.dispersion * log_lik/scale + log_prior - (self.dispersion* self.log_ref/scale)

    def langevin_sampler(self, nsample= 2000, nburnin=100, proposal_scale = np.identity, step=1.):

        state = self.initial_estimate
        stepsize = 1. / (step * self.ntarget)

        sampler = langevin(state, self.log_posterior, proposal_scale, stepsize)
        samples = np.zeros((nsample, self.ntarget))

        for i in range(nsample):
            sampler.next(scaling_ = self.dispersion)
            sys.stderr.write("sample number: " + str(i) + "sample: " + str(sampler.state.copy()) + "\n")
            samples[i, :] = sampler.state.copy()

        return samples[nburnin:, :]

    def gibbs_sampler(self, nsample= 2000, nburnin=100, proposal_scale = np.identity, step=1.):

        state = self.initial_estimate
        scale_state = self.dispersion
        stepsize = 1. /(step* self.ntarget)

        sampler = langevin(state, self.log_posterior, proposal_scale, stepsize)
        samples = np.zeros((nsample, self.ntarget))

        for i in range(nsample):
            sampler.next(scaling_= scale_state)
            scale_update = invgamma.rvs(a=(0.1 + self.ntarget + self.ntarget/2), scale=0.1 - (scale_state * sampler.grad_posterior[1]), size=1)

            scale_state = scale_update
            samples[i, :] = sampler.state.copy()
            sys.stderr.write("sample number: " + str(i) + "sample: " + str(samples[i, :]) + "\n")
            sys.stderr.write("sample number: " + str(i) + "sigma: " + str(scale_state) + "\n")

        return samples[nburnin:, :]


class langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 proposal_scale,
                 stepsize):

        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           stepsize)
        self.proposal_scale = proposal_scale
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0,scale=1)
        self.sample = np.copy(initial_condition)

    def __iter__(self):
        return self

    def next(self, scaling_):
        while True:
            self.grad_posterior = self.gradient_map(self.state, scaling_)
            candidate = (self.state + self.stepsize * self.proposal_scale.dot(self.grad_posterior[0])
                        + np.sqrt(2.)* (fractional_matrix_power(self.proposal_scale, 0.5).dot(self._noise.rvs(self._shape))) * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate)[0])):
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                break
