"""
Classes encapsulating some common workflows in randomized setting
"""

from copy import copy
import functools

import numpy as np
import regreg.api as rr

# from .glm import (glm_group_lasso,
#                   glm_group_lasso_parametric,
#                   glm_greedy_step,
#                   glm_threshold_score,
#                   glm_nonparametric_bootstrap,
#                   glm_parametric_covariance,
#                   pairs_bootstrap_glm)

from .randomization import randomization
from .query import multiple_queries, optimization_sampler
from .M_estimator import restricted_Mest

class lasso_iv(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \frac{1}{2n} \|y-X\beta\|^2_2 + 
            \lambda \|\beta\|_1 - \omega^T\beta + \frac{\epsilon}{2} \|\beta\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty.

    """

    def __init__(self, 
                 Y,
                 D, 
                 Z,
                 feature_weights,
                 ridge_term,
                 randomizer_scale,
                 randomizer='gaussian',
                 parametric_cov_estimator=False):
        r"""

        Create a new post-selection object for the LASSO problem

        Parameters
        ----------

        Y : response

        D : interest

        Z : instruments

        feature_weights : np.ndarray
            Feature weights for L-1 penalty for alpha. If a float,
            it is brodcast to all features.

        ridge_term : float
            How big a ridge term to add?

        randomizer_scale : float
            Scale for IID components of randomization.

        randomizer : str (optional)
            One of ['laplace', 'logistic', 'gaussian']


        """

        # form the projected design and response

        P_Z = Z.dot(np.linalg.pinv(Z))
        X = np.hstack([Z, D.reshape((-1,1))])
        P_ZX = P_Z.dot(X)
        P_ZY = P_Z.dot(Y)

        self.Z = Z
        self.P_ZX, self.P_ZY = P_ZX, P_ZY
        self.loglike = rr.glm.gaussian(P_ZX, P_ZY)
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(self.loglike.shape[0] - 1) * feature_weights
        self.feature_weights = np.hstack([np.asarray(feature_weights), 0])

        if randomizer == 'laplace':
            self.randomizer = randomization.laplace((p,), scale=randomizer_scale)
        elif randomizer == 'gaussian':
            self.randomizer = randomization.isotropic_gaussian((p,),randomizer_scale)
        elif randomizer == 'logistic':
            self.randomizer = randomization.logistic((p,), scale=randomizer_scale)

        self.ridge_term = ridge_term
        self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)
        self.loss = self.loglike


    def fit(self, solve_args={'tol':1.e-12, 'min_its':50}):
        """
        Fit the randomized lasso using `regreg`.

        Parameters
        ----------

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        views : list
             Other views of the data, e.g. cross-validation.

        Returns
        -------

        sign_beta : np.float
             Support and non-zero signs of randomized lasso solution.
             
        """

        p = self.nfeature

        # solve the optimization problem here, return the sign pattern

        # Randomization and quadratic term, e/2*\|x|_2^2, term setup (x = parameters penalized)
        random_linear_term = self.randomizer.sample()
        # rr.identity_quadratic essentially amounts to epsilon/2 * \|x - 0\|^2 + <-random_linear_term, x> + 0
        random_loss = rr.identity_quadratic(self.ridge_term, 0, -random_linear_term, 0)
        self.loss.quadratic = random_loss

        # Optimization problem   
        problem = rr.simple_problem(self.loss, self.penalty)
        problem.solve(**solve_args) 
        self.soln = problem.coefs

        self.signs = np.sign(self.soln)
        self.active_set = self.signs != 0
        self.inactive_set = ~self.active_set

        # for sampler

        self.inactive_weighted_sup = rr.weighted_supnorm(self.penalty.weights[self.inactive_set], bound=1.)
        self.active_slice = self.active_set.copy()
        self.active_slice[-1] = 0 # this is beta -- no constraint on sign
        self.active_signs = self.signs[self.active_set][:-1]
        self.subgrad_slice = self.inactive_set

        self.observed_opt_state = np.zeros_like(self.soln)
        self.observed_opt_state[self.active_set] = self.soln[self.active_set]
        self.observed_opt_state[self.inactive_set] = -(self.loss.smooth_objective(self.soln,'grad') +
                                                       self.loss.quadratic.objective(self.soln,'grad'))[self.inactive_set]

        return self.signs

    def get_sampler(self):
        # setup the default optimization sampler

        if not hasattr(self, "_sampler"):
            def projection(weighted_sup, subgrad_slice, active_slice, opt_state):
                """
                Full projection for Langevin.

                The state here will be only the state of the optimization variables.
                """

                new_state = opt_state.copy() # not really necessary to copy
                new_state[active_slice] = np.maximum(self.active_signs * opt_state[active_slice], 0) * self.active_signs
                new_state[subgrad_slice] = weighted_sup.bound_prox(opt_state[subgrad_slice])
                return new_state

            projection = functools.partial(projection, self.inactive_weighted_sup, self.subgrad_slice, self.active_slice)

            # X^TX matrix

            H = (self.P_ZX.T.dot(self.P_ZX) + np.identity(self.P_ZX.shape[1]) * self.ridge_term)[:,self.active_set]
            S = self.P_ZX.T.dot(self.P_ZY)

            #opt_linear = np.zeros(H.shape)
            opt_linear = np.zeros((H.shape[0], H.shape[0]))
            opt_linear[:, self.active_set] = H
            opt_linear[:, self.inactive_set][self.inactive_set] = np.identity(self.inactive_set.sum())

            opt_offset = np.zeros(opt_linear.shape[0])
            opt_offset[self.active_set] = self.feature_weights[self.active_set] * self.signs[self.active_set]

            opt_transform = opt_linear, opt_offset

            def grad_log_density(opt_transform,
                                 rand_gradient,
                                 S,
                                 opt_state):
                opt_linear, opt_offset = opt_transform
                opt_state = np.atleast_2d(opt_state)
                full_state = np.squeeze(opt_linear.dot(opt_state.T) + (opt_offset)[:, None]).T - S
                deriv = opt_linear.T.dot(rand_gradient(full_state).T)
                return deriv
                
            grad_log_density = functools.partial(grad_log_density, opt_transform, self.randomizer.gradient)

            def log_density(opt_transform,
                            rand_log_density,
                            S,
                            opt_state):
                opt_state = np.atleast_2d(opt_state)
                full_state = np.squeeze(opt_linear.dot(opt_state.T) + (opt_offset)[:, None]).T - S
                return rand_log_density(full_state)

            log_density = functools.partial(log_density, opt_transform, self.randomizer.log_density)

            self._sampler = optimization_sampler(self.observed_opt_state,
                                                 S,
                                                 (-np.identity(S.shape[0]), np.zeros(S.shape[0])),
                                                 opt_transform,
                                                 projection,
                                                 grad_log_density,
                                                 log_density)
        return self._sampler

    sampler = property(get_sampler)

    def summary(self,
                parameter=None,
                Sigma=1.,
                level=0.9,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=False):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.

        Sigma : true Sigma_11, known for now

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = np.zeros(self.loglike.shape[0])

        # compute tsls

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self.active_slice[:-1]].dot(np.linalg.pinv(self.Z[:,self.active_slice[:-1]]))
        P_ZD = self.P_ZX[:,-1]
        two_stage_ls = (P_ZD.dot(P_Z-P_ZE).dot(self.P_ZY-P_ZD*parameter))/np.sqrt(Sigma*P_ZD.dot(P_Z-P_ZE).dot(P_ZD))
        two_stage_ls = np.atleast_1d(two_stage_ls)
        target_cov = np.atleast_2d(1.)
        score_cov = -1.*np.sqrt(Sigma/P_ZD.dot(P_Z-P_ZE).dot(P_ZD))*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        score_cov = np.atleast_2d(score_cov)

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(two_stage_ls, target_cov, score_cov, parameter=parameter, sample=opt_sample)
        if np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(two_stage_ls, target_cov, score_cov, parameter=np.zeros_like(parameter), sample=opt_sample)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(two_stage_ls, target_cov, score_cov, sample=opt_sample)

        return pivots, pvalues, intervals
