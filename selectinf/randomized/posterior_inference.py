from __future__ import division, print_function

import numpy as np
import typing

from scipy.stats import norm as ndist, invgamma
from scipy.linalg import fractional_matrix_power

from ..algorithms.barrier_affine import solve_barrier_affine_py
from .selective_MLE import mle_inference
from ..base import target_query_Interactspec

class PosteriorAtt(typing.NamedTuple):

    logPosterior: float
    grad_logPosterior: np.ndarray

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
                 query_spec,
                 target_spec,
                 dispersion,
                 prior,
                 solve_args={'tol': 1.e-12}):

        self.query_spec = QS = query_spec
        self.target_spec = TS = target_spec
        self.solve_args = solve_args

        G = mle_inference(query_spec,
                          target_spec,
                          solve_args=solve_args)

        result, self.inverse_info, self.log_ref = G.solve_estimating_eqn()

        self.ntarget = TS.cov_target.shape[0]
        self.nopt = QS.cond_cov.shape[0]

        self.initial_estimate = np.asarray(result['MLE'])
        self.dispersion = dispersion

        ### Note for an informative prior we might want to change this...
        self.prior = prior

        self._get_marginal_parameters()

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

        QS = self.query_spec
        TS = self.target_spec
        
        (prec_marginal,
         linear_coef,
         offset_coef,
         r,
         S,
         prec_target_nosel) = self._get_marginal_parameters()
        
        sigmasq = sigma ** 2

        target = S.dot(target_parameter) + r

        mean_marginal = linear_coef.dot(target) + offset_coef
        conjugate_marginal = prec_marginal.dot(mean_marginal)

        solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_marginal,
                                 prec_marginal,
                                 QS.observed_soln,
                                 QS.linear_part,
                                 QS.offset,
                                 **self.solve_args)

        log_normalizer = -val - mean_marginal.T.dot(prec_marginal).dot(mean_marginal) / 2.

        log_lik = -((TS.observed_target - target).T.dot(prec_target_nosel).dot(TS.observed_target - target)) / 2. \
                  - log_normalizer

        grad_lik = S.T.dot(prec_target_nosel.dot(TS.observed_target) - prec_target_nosel.dot(target)
                                - linear_coef.T.dot(prec_marginal.dot(soln) - conjugate_marginal))

        log_prior, grad_prior = self.prior(target_parameter)

        log_posterior = self.dispersion * (log_lik - self.log_ref) / sigmasq + log_prior
        grad_log_posterior = self.dispersion * grad_lik / sigmasq + grad_prior

        return PosteriorAtt(log_posterior,
                            grad_log_posterior)

    ### Private method

    def _get_marginal_parameters(self):
        """
        This works out the implied covariance
        of optimization varibles as a function
        of randomization as well how to compute
        implied mean as a function of the true parameters.
        """

        QS = self.query_spec
        TS = self.target_spec

        U1, U2, U3, U4, U5 = target_query_Interactspec(QS,
                                                       TS.regress_target_score,
                                                       TS.cov_target)

        prec_target = np.linalg.inv(TS.cov_target)
        cond_precision = np.linalg.inv(QS.cond_cov)

        prec_target_nosel = prec_target + U2 - U3

        _P = -(U1.T.dot(QS.M1.dot(QS.observed_score)) + U2.dot(TS.observed_target))

        bias_target = TS.cov_target.dot(U1.T.dot(-U4.dot(TS.observed_target) +
                                                 QS.M1.dot(QS.opt_linear.dot(QS.cond_mean))) - _P)

        ###set parameters for the marginal distribution of optimization variables

        _Q = np.linalg.inv(prec_target_nosel + U3)
        prec_marginal = cond_precision - U5.T.dot(_Q).dot(U5)
        linear_coef = QS.cond_cov.dot(U5.T)
        offset_coef = QS.cond_mean - linear_coef.dot(TS.observed_target)

        ###set parameters for the marginal distribution of target

        r = np.linalg.inv(prec_target_nosel).dot(prec_target.dot(bias_target))
        S = np.linalg.inv(prec_target_nosel).dot(prec_target)

        return (prec_marginal,
                linear_coef,
                offset_coef,
                r,
                S,
                prec_target_nosel)

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

        import sys
        sys.stderr.write('a: ' + str(0.1 +
                          selective_posterior.ntarget +
                          selective_posterior.ntarget / 2)+'\n')
        sys.stderr.write('scale: ' + str(0.1 - ((scale_update ** 2) * sampler.posterior_[0])) + '\n')
        sys.stderr.write('scale_update: ' + str(scale_update) + '\n')
        sys.stderr.write('initpoint: ' + str(sampler.posterior_[0]) + '\n')
        scale_update_sq = invgamma.rvs(a=(0.1 +
                                          selective_posterior.ntarget +
                                          selective_posterior.ntarget / 2),
                                       scale=0.1 - ((scale_update ** 2) * sampler.posterior_.logPosterior),
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
            _proposal = self.proposal_sqrt.dot(self._noise.rvs(self._shape))
            candidate = (self.state + self.stepsize * self.proposal_scale.dot(self.posterior_.grad_logPosterior)
                         + np.sqrt(2.) * _proposal * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate, self.scaling)[1])):
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                break
        return self.state


def target_query_Interactspec(query_spec,
                              regress_target_score,
                              cov_target):

    QS = query_spec
    prec_target = np.linalg.inv(cov_target)

    U1 = regress_target_score.T.dot(prec_target)
    U2 = U1.T.dot(QS.M2.dot(U1))
    U3 = U1.T.dot(QS.M3.dot(U1))
    U4 = QS.M1.dot(QS.opt_linear).dot(QS.cond_cov).dot(QS.opt_linear.T.dot(QS.M1.T.dot(U1)))
    U5 = U1.T.dot(QS.M1.dot(QS.opt_linear))

    return U1, U2, U3, U4, U5
