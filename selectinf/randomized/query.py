import numpy as np
import pandas as pd
from scipy.stats import norm as ndist

from ..constraints.affine import constraints
from ..algorithms.barrier_affine import solve_barrier_affine_py

from .posterior_inference import posterior
from .approx_reference import approximate_grid_inference
from .exact_reference import exact_grid_inference

class query(object):
    r"""
    This class is the base of randomized selective inference
    based on convex programs.
    The main mechanism is to take an initial penalized program
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B)
    and add a randomization and small ridge term yielding
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B) -
        \langle \omega, B \rangle + \frac{\epsilon}{2} \|B\|^2_2
    """

    def __init__(self, randomization, perturb=None):

        """
        Parameters
        ----------
        randomization : `selection.randomized.randomization.randomization`
            Instance of a randomization scheme.
            Describes the law of $\omega$.
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """
        self.randomization = randomization
        self.perturb = perturb
        self._solved = False
        self._randomized = False
        self._setup = False

    # Methods reused by subclasses

    def randomize(self, perturb=None):

        """
        The actual randomization step.
        Parameters
        ----------
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """

        if not self._randomized:
            (self.randomized_loss,
             self._initial_omega) = self.randomization.randomize(self.loss,
                                                                 self.epsilon,
                                                                 perturb=perturb)
        self._randomized = True

    def get_sampler(self):
        if hasattr(self, "_sampler"):
            return self._sampler

    def set_sampler(self, sampler):
        self._sampler = sampler

    sampler = property(get_sampler, set_sampler, doc='Sampler of optimization (augmented) variables.')

    # implemented by subclasses

    def solve(self):

        raise NotImplementedError('abstract method')


class gaussian_query(query):

    """
    A class with Gaussian perturbation to the objective -- 
    easy to apply CLT to such things
    """

    def fit(self, perturb=None):

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

    # Private methods

    def _setup_sampler(self,
                       linear_part,
                       offset,
                       opt_linear,
                       observed_subgrad,
                       dispersion=1):

        A, b = linear_part, offset

        if not np.all(A.dot(self.observed_opt_state) - b <= 0):
            raise ValueError('constraints not satisfied')

        (cond_mean,
         cond_cov,
         cond_precision,
         regress_opt,
         M1,
         M2,
         M3) = self._setup_implied_gaussian(opt_linear,
                                            observed_subgrad,
                                            dispersion=dispersion)

        self.cond_mean, self.cond_cov = cond_mean, cond_cov

        affine_con = constraints(A,
                                 b,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        self.affine_con = affine_con
        self.opt_linear = opt_linear
        self.observed_subgrad = observed_subgrad

    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):

        cov_rand, prec = self.randomizer.cov_prec

        if np.asarray(prec).shape in [(), (0,)]:
            prod_score_prec_unnorm = self._unscaled_cov_score * prec
        else:
            prod_score_prec_unnorm = self._unscaled_cov_score.dot(prec)

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T) * prec
        else:
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T).dot(prec)

        # regress_opt is regression coefficient of opt onto score + u...

        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        M1 = prod_score_prec_unnorm * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T)

        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

        return (cond_mean,
                cond_cov,
                cond_precision,
                regress_opt,
                M1,
                M2,
                M3)

    def selective_MLE(self,
                      target_spec,
                      level=0.90,
                      solve_args={'tol': 1.e-12}):

        return selective_MLE(target_spec,
                             self.observed_opt_state,
                             self.affine_con.mean,
                             self.affine_con.covariance,
                             self.affine_con.linear_part,
                             self.affine_con.offset,
                             self.opt_linear,
                             self.M1,
                             self.M2,
                             self.M3,
                             self.observed_score_state + self.observed_subgrad,
                             solve_args=solve_args,
                             level=level,
                             useC=False)


    def posterior(self,
                  target_spec,
                  dispersion=1,
                  prior=None,
                  solve_args={'tol': 1.e-12}):
        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        prior : callable
            A callable object that takes a single argument
            `parameter` of the same shape as `observed_target`
            and returns (value of log prior, gradient of log prior)
        dispersion : float, optional
            Dispersion parameter for log-likelihood.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        if prior is None:
            Di = 1. / (200 * np.diag(target_spec.cov_target))

            def prior(target_parameter):
                grad_prior = -target_parameter * Di
                log_prior = -0.5 * np.sum(target_parameter ** 2 * Di)
                return log_prior, grad_prior

        return posterior(self,
                         target_spec,
                         dispersion,
                         prior,
                         solve_args=solve_args)

    def approximate_grid_inference(self,
                                   target_spec,
                                   useIP=True,
                                   solve_args={'tol': 1.e-12}):

        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        solve_args : dict, optional
            Arguments passed to solver.
        """

        G = approximate_grid_inference(self,
                                       target_spec,
                                       solve_args=solve_args,
                                       useIP=useIP)

        return G.summary(alternatives=target_spec.alternatives)

    def exact_grid_inference(self,
                             target_spec,
                             solve_args={'tol': 1.e-12}):

        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        solve_args : dict, optional
            Arguments passed to solver.
        """

        G = exact_grid_inference(self,
                                 target_spec,
                                 solve_args=solve_args)

        return G.summary(alternatives=target_spec.alternatives)


from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C

def selective_MLE(target_spec,
                  observed_soln,  # initial (observed) value of
                  # optimization variables -- used as a
                  # feasible point.  precise value used
                  # only for independent estimator
                  cond_mean,
                  cond_cov,
                  linear_part,
                  offset,
                  opt_linear,
                  M1,   
                  M2,
                  M3,
                  observed_score,
                  solve_args={'tol': 1.e-12},
                  level=0.9,
                  useC=False):

    """
    Selective MLE based on approximation of
    CGF.
    Parameters
    ----------
    observed_target : ndarray
        Observed estimate of target.
    cov_target : ndarray
        Estimated covaraince of target.
    regress_target_score : ndarray
        Estimated regression coefficient of target on score.
    observed_soln : ndarray
        Feasible point for optimization problem.
    cond_mean : ndarray
        Conditional mean of optimization variables given target.
    cond_cov : ndarray
        Conditional covariance of optimization variables given target.
    linear_part : ndarray
        Linear part of affine constraints: $\{o:Ao \leq b\}$
    offset : ndarray
        Offset part of affine constraints: $\{o:Ao \leq b\}$
    solve_args : dict, optional
        Arguments passed to solver.
    level : float, optional
        Confidence level.
    useC : bool, optional
        Use python or C solver.
    """

    (observed_target,
     cov_target,
     regress_target_score) = target_spec[:3]

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)

    prec_opt = np.linalg.inv(cond_cov)

    # this is specific to target
    
    T1 = regress_target_score.T.dot(prec_target)
    T2 = T1.T.dot(M2.dot(T1))
    T3 = T1.T.dot(M3.dot(T1)) 
    T4 = M1.dot(opt_linear).dot(cond_cov).dot(opt_linear.T.dot(M1.T.dot(T1)))
    T5 = T1.T.dot(M1.dot(opt_linear))

    prec_target_nosel = prec_target + T2 - T3

    _P = -(T1.T.dot(M1.dot(observed_score)) + T2.dot(observed_target)) ##flipped sign of second term here

    bias_target = cov_target.dot(T1.T.dot(-T4.dot(observed_target) + M1.dot(opt_linear.dot(cond_mean))) - _P)

    conjugate_arg = prec_opt.dot(cond_mean)

    if useC:
        solver = solve_barrier_affine_C
    else:
        solver = solve_barrier_affine_py

    val, soln, hess = solver(conjugate_arg,
                             prec_opt,
                             observed_soln,
                             linear_part,
                             offset,
                             **solve_args)

    final_estimator = cov_target.dot(prec_target_nosel).dot(observed_target) \
                      + regress_target_score.dot(M1.dot(opt_linear)).dot(cond_mean - soln) - bias_target

    observed_info_natural = prec_target_nosel + T3 - T5.dot(hess.dot(T5.T))

    unbiased_estimator = cov_target.dot(prec_target_nosel).dot(observed_target) - bias_target

    observed_info_mean = cov_target.dot(observed_info_natural.dot(cov_target))

    Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))

    pvalues = ndist.cdf(Z_scores)

    pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

    alpha = 1. - level

    quantile = ndist.ppf(1 - alpha / 2.)

    intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                           final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

    log_ref = val + conjugate_arg.T.dot(cond_cov).dot(conjugate_arg) / 2.

    result = pd.DataFrame({'MLE': final_estimator,
                           'SE': np.sqrt(np.diag(observed_info_mean)),
                           'Zvalue': Z_scores,
                           'pvalue': pvalues,
                           'lower_confidence': intervals[:, 0],
                           'upper_confidence': intervals[:, 1],
                           'unbiased': unbiased_estimator})

    return result, observed_info_mean, log_ref

