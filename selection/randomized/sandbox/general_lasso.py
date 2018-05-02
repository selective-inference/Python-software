from __future__ import print_function
import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr
import regreg.affine as ra

from ..constraints.affine import constraints
from ..algorithms.sqrt_lasso import solve_sqrt_lasso, choose_lambda

from .query import (query,
                    multiple_queries,
                    langevin_sampler,
                    affine_gaussian_sampler)

from .reconstruction import reconstruct_opt
from .randomization import randomization
from .base import restricted_estimator
from .glm import (pairs_bootstrap_glm,
                  glm_nonparametric_bootstrap,
                  glm_parametric_covariance)
from ..algorithms.debiased_lasso import debiasing_matrix


class lasso_view(query):
    def __init__(self,
                 loss,
                 epsilon,
                 penalty,
                 randomization,
                 perturb=None,
                 solve_args={'min_its': 50, 'tol': 1.e-10}):
        """
        Fits the logistic regression to a candidate active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.
        Computes $\bar{\beta}_E$ which is the restricted
        M-estimator (i.e. subject to the constraint $\beta_{-E}=0$).
        Parameters:
        -----------
        active: np.bool
            The active set from fitting the logistic lasso
        solve_args: dict
            Arguments to be passed to regreg solver.
        Returns:
        --------
        None
        Notes:
        ------
        Sets self._beta_unpenalized which will be used in the covariance matrix calculation.
        Also computes Hessian of loss at restricted M-estimator as well as the bootstrap covariance.
        """

        query.__init__(self, randomization)

        (self.loss,
         self.epsilon,
         self.penalty,
         self.randomization) = (loss,
                                epsilon,
                                penalty,
                                randomization)

    # Methods needed for subclassing a query

    def solve(self, nboot=2000,
              solve_args={'min_its': 20, 'tol': 1.e-10},
              perturb=None):

        self.randomize(perturb=perturb)

        (loss,
         randomized_loss,
         epsilon,
         penalty,
         randomization) = (self.loss,
                           self.randomized_loss,
                           self.epsilon,
                           self.penalty,
                           self.randomization)

        # initial solution

        p = penalty.shape[0]

        problem = rr.simple_problem(randomized_loss, penalty)
        self.initial_soln = problem.solve(**solve_args)

        # find the active groups and their direction vectors
        # as well as unpenalized groups

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        if isinstance(penalty, rr.l1norm):
            self._lagrange = penalty.lagrange * np.ones(p)
            unpenalized = np.zeros(p, np.bool)
        elif isinstance(penalty, rr.weighted_l1norm):
            self._lagrange = penalty.weights
            unpenalized = self._lagrange == 0
        else:
            raise ValueError('penalty must be `l1norm` or `weighted_l1norm`')

        active *= ~unpenalized

        # solve the restricted problem

        self._overall = (active + unpenalized) > 0
        self._inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan  # don't release sign of unpenalized variables
        self.selection_variable = {'sign': _active_signs,
                                   'variables': self._overall}

        # initial state for opt variables

        initial_subgrad = -(self.randomized_loss.smooth_objective(self.initial_soln, 'grad') +
                            self.randomized_loss.quadratic.objective(self.initial_soln, 'grad'))
        # the quadratic of a smooth_atom is not included in computing the smooth_objective
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.fabs(self.initial_soln[active])
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  self.initial_subgrad[self._inactive]], axis=0)

        # set the _solved bit

        self._solved = True

        # Now setup the pieces for linear decomposition

        (loss,
         epsilon,
         penalty,
         initial_soln,
         overall,
         inactive,
         unpenalized) = (self.loss,
                         self.epsilon,
                         self.penalty,
                         self.initial_soln,
                         self._overall,
                         self._inactive,
                         self._unpenalized)

        # we are implicitly assuming that
        # loss is a pairs model

        _beta_unpenalized = restricted_estimator(loss, overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # observed state for score in internal coordinates

        self.observed_internal_state = np.hstack([_beta_unpenalized,
                                                  -loss.smooth_objective(beta_bar, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, p))
        _score_linear_term = np.zeros((p, p))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        est_slice = slice(0, overall.sum())
        X, y = loss.data
        W = self.loss.saturated_loss.hessian(X.dot(beta_bar))
        _hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _hessian_unpen = np.dot(X.T, X[:, unpenalized] * W[:, None])

        _score_linear_term[:, est_slice] = -np.hstack([_hessian_active, _hessian_unpen])

        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        null_idx = np.arange(overall.sum(), p)
        inactive_idx = np.nonzero(inactive)[0]
        for _i, _n in zip(inactive_idx, null_idx):
            _score_linear_term[_i, _n] = -1

        # c_E piece

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p, j, active_signs[j]) for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = _hessian_active * active_signs[None, active] + epsilon * active_directions
        _opt_linear_term[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), active.sum() + unpenalized.sum())
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (_hessian_unpen
                                                      + epsilon * unpenalized_directions)

            # subgrad piece

        subgrad_idx = range(active.sum() + unpenalized.sum(), active.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active.sum() + unpenalized.sum(), active.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i, _s] = 1

        # form affine part

        _opt_affine_term = np.zeros(p)
        idx = 0
        _opt_affine_term[active] = active_signs[active] * self._lagrange[active]

        # two transforms that encode score and optimization
        # variable roles

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # everything now expressed in observed_score_state

        self.observed_score_state = _score_linear_term.dot(self.observed_internal_state)

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        # we form a dual group lasso object
        # to do the projection


        self._setup = True
        self.subgrad_slice = subgrad_slice
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.ndim = loss.shape[0]

        self.nboot = nboot

    def get_sampler(self):
        # setup the default optimization sampler

        if not hasattr(self, "_sampler"):

            penalty, inactive = self.penalty, self._inactive
            inactive_lagrange = self.penalty.weights[inactive]

            if not hasattr(self.randomization, "cov_prec"):  # means randomization is not Gaussian

                dual = rr.weighted_supnorm(1. / inactive_lagrange, bound=1.)

                def projection(dual, subgrad_slice, scaling_slice, opt_state):
                    """
                    Full projection for Langevin.
                    The state here will be only the state of the optimization variables.
                    """

                    new_state = opt_state.copy()  # not really necessary to copy
                    new_state[scaling_slice] = np.maximum(opt_state[scaling_slice], 0)
                    new_state[subgrad_slice] = dual.bound_prox(opt_state[subgrad_slice])
                    return new_state

                projection = functools.partial(projection, dual, self.subgrad_slice, self.scaling_slice)

                def grad_log_density(query,
                                     rand_gradient,
                                     score_state,
                                     opt_state):
                    full_state = score_state + reconstruct_opt(query.opt_transform, opt_state)
                    return opt_linear.T.dot(rand_gradient(full_state).T)

                grad_log_density = functools.partial(grad_log_density, self, self.randomization.gradient)

                def log_density(query,
                                opt_linear,
                                rand_log_density,
                                score_state,
                                opt_state):
                    full_state = score_state + reconstruct_opt(query.opt_transform, opt_state)
                    return rand_log_density(full_state)

                log_density = functools.partial(log_density, self, self.randomization.log_density)

                self._sampler = langevin_sampler(self.observed_opt_state,
                                                 self.observed_score_state,
                                                 self.score_transform,
                                                 self.opt_transform,
                                                 projection,
                                                 grad_log_density,
                                                 log_density)
            else:

                # compute implied mean and covariance

                cov, prec = self.randomization.cov_prec
                prec_array = len(np.asarray(prec).shape) == 2
                opt_linear, opt_offset = self.opt_transform

                if prec_array:
                    cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
                    cond_cov = np.linalg.inv(cond_precision)
                    logdens_linear = cond_cov.dot(opt_linear.T.dot(prec))
                else:
                    cond_precision = opt_linear.T.dot(opt_linear) * prec
                    cond_cov = np.linalg.inv(cond_precision)
                    logdens_linear = cond_cov.dot(opt_linear.T) * prec

                cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

                # need a log_density function
                # the conditional density of opt variables
                # given the score

                def log_density(logdens_linear, offset, cond_prec, score, opt):
                    if score.ndim == 1:
                        mean_term = logdens_linear.dot(score.T + offset).T
                    else:
                        mean_term = logdens_linear.dot(score.T + offset[:, None]).T
                    arg = opt + mean_term
                    return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

                log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

                # now make the constraints

                # scaling constraints

                I = np.identity(cond_cov.shape[0])
                A_scaling = -I[self.scaling_slice]
                b_scaling = np.zeros(A_scaling.shape[0])

                A_subgrad = np.vstack([I[self.subgrad_slice],
                                       -I[self.subgrad_slice]])
                b_subgrad = np.hstack([inactive_lagrange,
                                       inactive_lagrange])

                linear_term = np.vstack([A_scaling, A_subgrad])
                offset = np.hstack([b_scaling, b_subgrad])

                affine_con = constraints(linear_term,
                                         offset,
                                         mean=cond_mean,
                                         covariance=cond_cov)

                logdens_transform = (logdens_linear, opt_offset)

                self._sampler = affine_gaussian_sampler(affine_con,
                                                        self.observed_opt_state,
                                                        self.observed_score_state,
                                                        log_density,
                                                        logdens_transform,
                                                        selection_info=self.selection_variable)  # should be signs and the subgradients we've conditioned on

        return self._sampler

    sampler = property(get_sampler, query.set_sampler)

    def decompose_subgradient(self, condition=None, marginalize=None):
        """
        ADD DOCSTRING
        condition and marginalize should be disjoint
        """

        p = self.penalty.shape[0]
        condition_inactive = np.zeros(p, dtype=np.bool)

        if condition is None:
            condition = np.zeros(p, dtype=np.bool)

        if marginalize is None:
            marginalize = np.zeros(p, dtype=np.bool)
            marginalize[self._overall] = 0

        if np.any(condition * marginalize):
            raise ValueError("cannot simultaneously condition and marginalize over a group's subgradient")

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        _inactive = self._inactive

        limits_marginal = np.zeros_like(_inactive, np.float)

        condition_inactive = _inactive * condition
        moving_inactive = _inactive * ~(marginalize + condition)
        margin_inactive = _inactive * marginalize

        limits_marginal = self._lagrange
        if np.asarray(self._lagrange).shape in [(), (1,)]:
            limits_marginal = np.zeros_like(_inactive) * self._lagrange

        opt_linear, opt_offset = self.opt_transform

        new_linear = np.zeros((opt_linear.shape[0], (self._active.sum() +
                                                     self._unpenalized.sum() +
                                                     moving_inactive.sum())))
        new_linear[:, self.scaling_slice] = opt_linear[:, self.scaling_slice]
        new_linear[:, self.unpenalized_slice] = opt_linear[:, self.unpenalized_slice]

        inactive_moving_idx = np.nonzero(moving_inactive)[0]
        subgrad_idx = range(self._active.sum() + self._unpenalized.sum(),
                            self._active.sum() + self._unpenalized.sum() +
                            moving_inactive.sum())
        for _i, _s in zip(inactive_moving_idx, subgrad_idx):
            new_linear[_i, _s] = 1.

        observed_opt_state = self.observed_opt_state[:(self._active.sum() +
                                                       self._unpenalized.sum() +
                                                       moving_inactive.sum())]
        observed_opt_state[subgrad_idx] = self.initial_subgrad[moving_inactive]

        condition_linear = np.zeros((opt_linear.shape[0], (self._active.sum() +
                                                           self._unpenalized.sum() +
                                                           condition_inactive.sum())))

        new_offset = opt_offset + 0.
        new_offset[condition_inactive] += self.initial_subgrad[condition_inactive]
        new_opt_transform = (new_linear, new_offset)

        if not hasattr(self.randomization, "cov_prec") or marginalize.sum():  # use Langevin -- not gaussian

            def _fraction(_cdf, _pdf, full_state_plus, full_state_minus, margin_inactive):
                return (np.divide(_pdf(full_state_plus) - _pdf(full_state_minus),
                                  _cdf(full_state_plus) - _cdf(full_state_minus)))[margin_inactive]

            def new_grad_log_density(query,
                                     limits_marginal,
                                     margin_inactive,
                                     _cdf,
                                     _pdf,
                                     new_opt_transform,
                                     deriv_log_dens,
                                     score_state,
                                     opt_state):

                full_state = score_state + reconstruct_opt(new_opt_transform, opt_state)

                p = query.penalty.shape[0]
                weights = np.zeros(p)

                if margin_inactive.sum() > 0:
                    full_state_plus = full_state + limits_marginal * margin_inactive
                    full_state_minus = full_state - limits_marginal * margin_inactive
                    weights[margin_inactive] = _fraction(_cdf, _pdf, full_state_plus, full_state_minus, margin_inactive)
                weights[~margin_inactive] = deriv_log_dens(full_state)[~margin_inactive]
                return -opt_linear.T.dot(weights)

            new_grad_log_density = functools.partial(new_grad_log_density,
                                                     self,
                                                     limits_marginal,
                                                     margin_inactive,
                                                     self.randomization._cdf,
                                                     self.randomization._pdf,
                                                     new_opt_transform,
                                                     self.randomization._derivative_log_density)

            def new_log_density(query,
                                limits_marginal,
                                margin_inactive,
                                _cdf,
                                _pdf,
                                new_opt_transform,
                                log_dens,
                                score_state,
                                opt_state):

                full_state = score_state + reconstruct_opt(new_opt_transform, opt_state)

                full_state = np.atleast_2d(full_state)
                p = query.penalty.shape[0]
                logdens = np.zeros(full_state.shape[0])

                if margin_inactive.sum() > 0:
                    full_state_plus = full_state + limits_marginal * margin_inactive
                    full_state_minus = full_state - limits_marginal * margin_inactive
                    logdens += np.sum(np.log(_cdf(full_state_plus) - _cdf(full_state_minus))[:, margin_inactive],
                                      axis=1)

                logdens += log_dens(full_state[:, ~margin_inactive])

                return np.squeeze(logdens)  # should this be negative to match the gradient log density?

            new_log_density = functools.partial(new_log_density,
                                                self,
                                                limits_marginal,
                                                margin_inactive,
                                                self.randomization._cdf,
                                                self.randomization._pdf,
                                                new_opt_transform,
                                                self.randomization._log_density)

            new_lagrange = self.penalty.weights[moving_inactive]
            new_dual = rr.weighted_l1norm(new_lagrange, lagrange=1.).conjugate

            def new_projection(dual,
                               noverall,
                               opt_state):
                new_state = opt_state.copy()
                new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
                new_state[noverall:] = dual.bound_prox(opt_state[noverall:])
                return new_state

            new_projection = functools.partial(new_projection,
                                               new_dual,
                                               self._overall.sum())

            new_selection_variable = copy(self.selection_variable)
            new_selection_variable['subgradient'] = self.observed_opt_state[condition_inactive]

            self.sampler = langevin_sampler(observed_opt_state,
                                            self.observed_score_state,
                                            self.score_transform,
                                            new_opt_transform,
                                            new_projection,
                                            new_grad_log_density,
                                            new_log_density,
                                            selection_info=(self, new_selection_variable))
        else:

            cov, prec = self.randomization.cov_prec
            prec_array = len(np.asarray(prec).shape) == 2

            if prec_array:
                cond_precision = new_linear.T.dot(prec.dot(new_linear))
                cond_cov = np.linalg.inv(cond_precision)
                logdens_linear = cond_cov.dot(new_linear.T.dot(prec))
            else:
                cond_precision = new_linear.T.dot(new_linear) * prec
                cond_cov = np.linalg.inv(cond_precision)
                logdens_linear = cond_cov.dot(new_linear.T) * prec

            cond_mean = -logdens_linear.dot(self.observed_score_state + new_offset)

            def log_density(logdens_linear, offset, cond_prec, score, opt):
                if score.ndim == 1:
                    mean_term = logdens_linear.dot(score.T + offset).T
                else:
                    mean_term = logdens_linear.dot(score.T + offset[:, None]).T
                arg = opt + mean_term
                return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

            log_density = functools.partial(log_density, logdens_linear, new_offset, cond_precision)

            # now make the constraints

            # scaling constraints

            # the scalings are first set of opt variables
            # then unpenalized
            # then the subgradients

            I = np.identity(cond_cov.shape[0])
            A_scaling = -I[self.scaling_slice]
            b_scaling = np.zeros(A_scaling.shape[0])

            A_subgrad = np.vstack([I[self._overall.sum():],
                                   -I[self._overall.sum():]])

            inactive_lagrange = self.penalty.weights[moving_inactive]
            b_subgrad = np.hstack([inactive_lagrange,
                                   inactive_lagrange])

            linear_term = np.vstack([A_scaling, A_subgrad])
            offset = np.hstack([b_scaling, b_subgrad])

            affine_con = constraints(linear_term,
                                     offset,
                                     mean=cond_mean,
                                     covariance=cond_cov)

            logdens_transform = (logdens_linear, new_offset)
            self._sampler = affine_gaussian_sampler(affine_con,
                                                    observed_opt_state,
                                                    self.observed_score_state,
                                                    log_density,
                                                    logdens_transform,
                                                    selection_info=self.selection_variable)  # should be signs and the subgradients we've conditioned on


class glm_lasso(lasso_view):
    def setup_sampler(self, scaling=1., solve_args={'min_its': 50, 'tol': 1.e-10}):
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self.selection_variable['variables'],
                                              beta_full=self._beta_full,
                                              inactive=~self.selection_variable['variables'])[0]

        return bootstrap_score


class glm_lasso_parametric(lasso_view):
    # this setup_sampler returns only the active set

    def setup_sampler(self):
        return self.selection_variable['variables']


class fixedX_lasso(lasso_view):
    def __init__(self, X, Y, epsilon, penalty, randomization, solve_args={'min_its': 50, 'tol': 1.e-10}):
        loss = glm.gaussian(X, Y)
        lasso_view.__init__(self,
                            loss,
                            epsilon,
                            penalty,
                            randomization,
                            solve_args=solve_args)

    def setup_sampler(self):
        X, Y = self.loss.data

        bootstrap_score = resid_bootstrap(self.loss,
                                          self.selection_variable['variables'],
                                          ~self.selection_variable['variables'])[0]
        return bootstrap_score


##### The class for users

class lasso(object):
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
                 loglike,
                 feature_weights,
                 ridge_term,
                 randomizer_scale,
                 randomizer='gaussian',
                 parametric_cov_estimator=False,
                 perturb=None):
        r"""
        Create a new post-selection object for the LASSO problem
        Parameters
        ----------
        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.
        feature_weights : np.ndarray
            Feature weights for L-1 penalty. If a float,
            it is brodcast to all features.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomization.
        randomizer : str (optional)
            One of ['laplace', 'logistic', 'gaussian']
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        self.parametric_cov_estimator = parametric_cov_estimator

        if randomizer == 'laplace':
            self.randomizer = randomization.laplace((p,), scale=randomizer_scale)
        elif randomizer == 'gaussian':
            self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        elif randomizer == 'logistic':
            self.randomizer = randomization.logistic((p,), scale=randomizer_scale)

        self.ridge_term = ridge_term

        self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)

        self._initial_omega = perturb

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None,
            nboot=1000):
        """
        Fit the randomized lasso using `regreg`.
        Parameters
        ----------
        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.
        Returns
        -------
        signs : np.float
             Support and non-zero signs of randomized lasso solution.

        """

        if perturb is not None:
            self._initial_omega = perturb

        p = self.nfeature
        if self.parametric_cov_estimator == True:
            self._view = glm_lasso_parametric(self.loglike, self.ridge_term, self.penalty, self.randomizer)
        else:
            self._view = glm_lasso(self.loglike, self.ridge_term, self.penalty, self.randomizer)
        self._view.solve(nboot=nboot, perturb=self._initial_omega, solve_args=solve_args)

        self.signs = np.sign(self._view.initial_soln)
        self.selection_variable = self._view.selection_variable
        return self.signs

    def decompose_subgradient(self,
                              condition=None,
                              marginalize=None):
        """
        Marginalize over some if inactive part of subgradient
        if applicable.
        Parameters
        ----------
        condition : np.bool
             Which groups' subgradients should we condition on.
        marginalize : np.bool
             Which groups' subgradients should we marginalize over.
        Returns
        -------
        None
        """

        if not hasattr(self, "_view"):
            raise ValueError("fit method should be run first")
        self._view.decompose_subgradient(condition=condition,
                                         marginalize=marginalize)

    def summary(self,
                selected_features,
                parameter=None,
                level=0.9,
                ndraw=10000,
                burnin=2000,
                compute_intervals=False,
                bootstrap_sampler=False,
                subset=None):
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
        level : float
            Confidence level.
        ndraw : int (optional)
            Defaults to 1000.
        burnin : int (optional)
            Defaults to 1000.
        bootstrap : bool
            Use wild bootstrap instead of Gaussian plugin.
        """
        if not hasattr(self, "_view"):
            raise ValueError('run `fit` method before producing summary.')

        if parameter is None:
            parameter = np.zeros(self.loglike.shape[0])

        if np.asarray(selected_features).dtype != np.bool:
            raise ValueError('selected_features should be a boolean array')

        unpenalized_mle = restricted_estimator(self.loglike, selected_features)

        if self.parametric_cov_estimator == False:
            n = self.loglike.data[0].shape[0]
            form_covariances = glm_nonparametric_bootstrap(n, n)
            boot_target, boot_target_observed = pairs_bootstrap_glm(self.loglike, selected_features, inactive=None)
            target_info = boot_target
        else:
            target_info = (selected_features, np.identity(unpenalized_mle.shape[0]))
            form_covariances = glm_parametric_covariance(self.loglike)

        opt_samplers = []
        for q in [self._view]:
            cov_info = q.setup_sampler()
            if self.parametric_cov_estimator == False:
                target_cov, score_cov = form_covariances(target_info,
                                                         cross_terms=[cov_info],
                                                         nsample=q.nboot)
            else:
                target_cov, score_cov = form_covariances(target_info,
                                                         cross_terms=[cov_info])
            opt_samplers.append(q.sampler)

        opt_samples = [opt_sampler.sample(ndraw,
                                          burnin) for opt_sampler in opt_samplers]

        if subset is not None:
            target_cov = target_cov[subset][:, subset]
            score_cov = score_cov[subset]
            unpenalized_mle = unpenalized_mle[subset]

        pivots = opt_samplers[0].coefficient_pvalues(unpenalized_mle, target_cov, score_cov, parameter=parameter,
                                                     sample=opt_samples[0])
        if not np.all(parameter == 0):
            pvalues = opt_samplers[0].coefficient_pvalues(unpenalized_mle, target_cov, score_cov,
                                                          parameter=np.zeros_like(parameter), sample=opt_samples[0])
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = opt_samplers[0].confidence_intervals(unpenalized_mle, target_cov, score_cov,
                                                             sample=opt_samples[0])

        return pivots, pvalues, intervals

    @staticmethod
    def gaussian(X,
                 Y,
                 feature_weights,
                 sigma=1.,
                 parametric_cov_estimator=False,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None,
                 randomizer='gaussian',
                 perturb=None):
        r"""
        Squared-error LASSO with feature weights.
        Objective function (before randomizer) is
        $$
        \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\lambda$ is `feature_weights`. The ridge term
        is determined by the Hessian and `np.std(Y)` (scaled by $\sqrt{n/(n-1)}$) by default,
        as is the randomizer scale.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        Y : ndarray
            Shape (n,) -- the response.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        sigma : float (optional)
            Noise variance. Set to 1 if `covariance_estimator` is not None.
            This scales the loglikelihood by `sigma**(-2)`.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        return lasso(loglike,
                     np.asarray(feature_weights) / sigma ** 2,
                     ridge_term,
                     randomizer_scale,
                     randomizer=randomizer,
                     parametric_cov_estimator=parametric_cov_estimator,
                     perturb=perturb)

    @staticmethod
    def logistic(X,
                 successes,
                 feature_weights,
                 trials=None,
                 parametric_cov_estimator=False,
                 quadratic=None,
                 ridge_term=None,
                 randomizer='gaussian',
                 randomizer_scale=None,
                 perturb=None):
        r"""
        Logistic LASSO with feature weights.
        Objective function is
        $$
        \beta \mapsto \ell(X\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\ell$ is the negative of the logistic
        log-likelihood (half the logistic deviance)
        and $\lambda$ is `feature_weights`.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        successes : ndarray
            Shape (n,) -- response vector. An integer number of successes.
            For data that is proportions, multiply the proportions
            by the number of trials first.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        trials : ndarray (optional)
            Number of trials per response, defaults to
            ones the same shape as Y.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        n, p = X.shape

        loglike = rr.glm.logistic(X, successes, trials=trials, quadratic=quadratic)

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5

        return lasso(loglike, feature_weights,
                     ridge_term,
                     randomizer_scale,
                     parametric_cov_estimator=parametric_cov_estimator,
                     randomizer=randomizer,
                     perturb=perturb)

    @staticmethod
    def coxph(X,
              times,
              status,
              feature_weights,
              parametric_cov_estimator=False,
              quadratic=None,
              ridge_term=None,
              randomizer='gaussian',
              randomizer_scale=None,
              perturb=None):
        r"""
        Cox proportional hazards LASSO with feature weights.
        Objective function is
        $$
        \beta \mapsto \ell^{\text{Cox}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\ell^{\text{Cox}}$ is the
        negative of the log of the Cox partial
        likelihood and $\lambda$ is `feature_weights`.
        Uses Efron's tie breaking method.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        times : ndarray
            Shape (n,) -- the survival times.
        status : ndarray
            Shape (n,) -- the censoring status.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        covariance_estimator : optional
            If None, use the parameteric
            covariance estimate of the selected model.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        loglike = coxph_obj(X, times, status, quadratic=quadratic)

        # scale for randomization seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(times) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer_scale,
                     randomizer=randomizer,
                     parametric_cov_estimator=parametric_cov_estimator,
                     perturb=perturb)

    @staticmethod
    def poisson(X,
                counts,
                feature_weights,
                parametric_cov_estimator=False,
                quadratic=None,
                ridge_term=None,
                randomizer_scale=None,
                randomizer='gaussian',
                perturb=None):
        r"""
        Poisson log-linear LASSO with feature weights.
        Objective function is
        $$
        \beta \mapsto \ell^{\text{Poisson}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\ell^{\text{Poisson}}$ is the negative
        of the log of the Poisson likelihood (half the deviance)
        and $\lambda$ is `feature_weights`.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        counts : ndarray
            Shape (n,) -- the response.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        randomizer : str
            One of ['laplace', 'logistic', 'gaussian']
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        """
        n, p = X.shape
        loglike = rr.glm.poisson(X, counts, quadratic=quadratic)

        # scale for randomizer seems kind of meaningless here...

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.std(counts) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(counts) * np.sqrt(n / (n - 1.))

        return lasso(loglike,
                     feature_weights,
                     ridge_term,
                     randomizer_scale,
                     randomizer=randomizer,
                     parametric_cov_estimator=parametric_cov_estimator,
                     perturb=perturb)

    @staticmethod
    def sqrt_lasso(X,
                   Y,
                   feature_weights,
                   quadratic=None,
                   parametric_cov_estimator=False,
                   sigma_estimate='truncated',
                   solve_args={'min_its': 200},
                   randomizer_scale=None,
                   perturb=None):
        r"""
        Use sqrt-LASSO to choose variables.
        Objective function is
        $$
        \beta \mapsto \|Y-X\beta\|_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        $$
        where $\lambda$ is `feature_weights`. After solving the problem
        treat as if `gaussian` with implied variance and choice of
        multiplier. See arxiv.org/abs/1504.08031 for details.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        Y : ndarray
            Shape (n,) -- the response.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        covariance : str
            One of 'parametric' or 'sandwich'. Method
            used to estimate covariance for inference
            in second stage.
        sigma_estimate : str
            One of 'truncated' or 'OLS'. Method
            used to estimate $\sigma$ when using
            parametric covariance.
        solve_args : dict
            Arguments passed to solver.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        Returns
        -------
        L : `selection.randomized.convenience.lasso`

        Notes
        -----
        Unlike other variants of LASSO, this
        solves the problem on construction as the active
        set is needed to find equivalent gaussian LASSO.
        Assumes parametric model is correct for inference,
        i.e. does not accept a covariance estimator.
        """

        n, p = X.shape

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.sqrt(n / (n - 1.))

        if perturb is None:
            perturb = np.random.standard_normal(p) * randomizer_scale

        randomQ = rr.identity_quadratic(ridge_term, 0, -perturb, 0)  # a ridge + linear term

        if quadratic is not None:
            totalQ = randomQ + quadratic
        else:
            totalQ = randomQ

        soln, sqrt_loss = solve_sqrt_lasso(X,
                                           Y,
                                           weights=feature_weights,
                                           quadratic=totalQ,
                                           solve_args=solve_args,
                                           force_fat=True)

        denom = np.linalg.norm(Y - X.dot(soln))

        loglike = rr.glm.gaussian(X, Y)

        raise NotImplementedError(
            'lasso_view needs to be modified so that the initial randomization can be set at construction time')

        return lasso(loglike,
                     np.asarray(feature_weights) * denom,
                     ridge_term * denom,
                     randomizer_scale * denom,
                     randomizer='gaussian',
                     parametric_cov_estimator=parametric_cov_estimator,
                     perturb=perturb)


