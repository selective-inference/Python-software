import numpy as np, pandas as pd

from ..constraints.affine import constraints
from .posterior_inference import (posterior, langevin_sampler)
from .approx_reference import approximate_grid_inference
from .exact_reference import exact_grid_inference
from .selective_MLE import mle_inference

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
                M1,
                M2,
                M3)

    def inference(self,
                  target_spec,
                  method,
                  level=0.90,
                  method_args={}):

        """
        Parameters
        ----------
        target_spec : TargetSpec
           Information needed to specify the target.
        method : str
           One of ['selective_MLE', 'approx', 'exact', 'posterior']
        level : float
           Confidence level or posterior quantiles.
        method_args : dict
           Dict of arguments to be optionally passed to the methods.

        Returns
        -------

        summary : pd.DataFrame
           Statistical summary for specified targets.
        """

        if method == 'selective_MLE':
            return self._selective_MLE(target_spec,
                                       level=level,
                                       **method_args)[0]
        elif method == 'exact':
            return self._exact_grid_inference(target_spec,
                                              level=level) # has no additional args
        elif method == 'approx':
            return self._approximate_grid_inference(target_spec,
                                                    level=level,
                                                    **method_args)
        elif method == 'posterior':
            return self.posterior(target_spec,
                                  **method_args)[1]

                                              
    def posterior(self,
                  target_spec,
                  level=0.90,
                  dispersion=1,
                  prior=None,
                  solve_args={'tol': 1.e-12},
                  nsample=2000,
                  nburnin=500):
        """

        Parameters
        ----------
        target_spec : TargetSpec
            Information needed to specify the target.
        level : float
            Level for credible interval.
        dispersion : float, optional
            Dispersion parameter for log-likelihood.
        prior : callable
            A callable object that takes a single argument
            `parameter` of the same shape as `observed_target`
            and returns (value of log prior, gradient of log prior)
        solve_args : dict, optional
            Arguments passed to solver.

        """

        if prior is None:
            Di = 1. / (200 * np.diag(target_spec.cov_target))

            def prior(target_parameter):
                grad_prior = -target_parameter * Di
                log_prior = -0.5 * np.sum(target_parameter ** 2 * Di)
                return log_prior, grad_prior

        posterior_repr =  posterior(self,
                                    target_spec,
                                    dispersion,
                                    prior,
                                    solve_args=solve_args)
        
        samples = langevin_sampler(posterior_repr,
                                   nsample=nsample,
                                   nburnin=nburnin)

        delta = 0.5 * (1 - level) * 100
        lower = np.percentile(samples, delta, axis=0)
        upper = np.percentile(samples, 100 - delta, axis=0)
        mean = np.mean(samples, axis=0)

        return samples, pd.DataFrame({'estimate':mean,
                                      'lower_credible':lower,
                                      'upper_credible':upper})
        
    # private methods

    def _selective_MLE(self,
                       target_spec,
                       level=0.90,
                       solve_args={'tol': 1.e-12}):

        """
        Parameters
        ----------
        target_spec : TargetSpec
           Information needed to specify the target.
        level : float
           Confidence level or posterior quantiles.
        solve_args : dict
           Dict of arguments to be optionally passed to solver.
        """

        G = mle_inference(self,
                          target_spec,
                          solve_args=solve_args)

        return G.solve_estimating_eqn(level=level)


    def _approximate_grid_inference(self,
                                    target_spec,
                                    level=0.90,
                                    solve_args={'tol': 1.e-12},
                                    useIP=True):

        """
        Parameters
        ----------
        target_spec : TargetSpec
           Information needed to specify the target.
        level : float
           Confidence level or posterior quantiles.
        solve_args : dict, optional
            Arguments passed to solver.
        useIP : bool
           Use spline extrapolation.
        """

        G = approximate_grid_inference(self,
                                       target_spec,
                                       solve_args=solve_args,
                                       useIP=useIP)

        return G.summary(alternatives=target_spec.alternatives,
                         level=level)

    def _exact_grid_inference(self,
                              target_spec,
                              level=0.90,
                              solve_args={'tol': 1.e-12}):

        """
        Parameters
        ----------
        target_spec : TargetSpec
           Information needed to specify the target.
        level : float
           Confidence level or posterior quantiles.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        G = exact_grid_inference(self,
                                 target_spec)

        return G.summary(alternatives=target_spec.alternatives,
                         level=level)



