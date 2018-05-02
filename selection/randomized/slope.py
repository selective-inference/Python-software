from __future__ import print_function

import functools
import numpy as np

# sklearn imports

have_isotonic = False
try:
    from sklearn.isotonic import IsotonicRegression
    have_isotonic = True
except ImportError:
    raise ValueError('unable to import isotonic regression from sklearn, SLOPE subgradient projection will not work')

# regreg imports

from regreg.atoms.slope import _basic_proximal_map
import regreg.api as rr

from ..constraints.affine import constraints

from .randomization import randomization
from .base import restricted_estimator
from .lasso import lasso
from .query import (query,
                    multiple_queries,
                    langevin_sampler,
                    affine_gaussian_sampler)

class slope(lasso):

    def __init__(self,
                 loglike,
                 slope_weights,
                 ridge_term,
                 randomizer_scale,
                 perturb=None):
        r"""
        Create a new post-selection object for the SLOPE problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        slope_weights : np.ndarray
            SLOPE weights for L-1 penalty. If a float,
            it is broadcast to all features.

        ridge_term : float
            How big a ridge term to add?

        randomizer_scale : float
            Scale for IID components of randomization.

        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(slope_weights).shape == ():
            slope_weights = np.ones(loglike.shape) * slope_weights
        self.slope_weights = np.asarray(slope_weights)

        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        self.ridge_term = ridge_term
        self.penalty = rr.slope(slope_weights, lagrange=1.)
        self._initial_omega = perturb  # random perturbation

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        self.initial_soln = problem.solve(quad, **solve_args)

        # now we have to work out SLOPE details, clusters, etc.

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        self._overall = overall = active> 0
        self._inactive = inactive = ~self._overall

        _active_signs = active_signs.copy()
        self.selection_variable = {'sign': _active_signs,
                                   'variables': self._overall}

        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') +
                            quad.objective(self.initial_soln, 'grad'))
        self.initial_subgrad = initial_subgrad

        indices = np.argsort(-np.fabs(self.initial_soln))
        sorted_soln = self.initial_soln[indices]
        initial_scalings = np.sort(np.unique(np.fabs(self.initial_soln[active])))[::-1]
        self.observed_opt_state = initial_scalings
        self._unpenalized = np.zeros(p, np.bool)

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        self.num_opt_var = self.observed_opt_state.shape[0]

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        _hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _score_linear_term = -_hessian_active
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        cur_indx_array = []
        cur_indx_array.append(0)
        cur_indx = 0
        pointer = 0
        signs_cluster = []
        for j in range(p - 1):
            if np.abs(sorted_soln[j + 1]) != np.abs(sorted_soln[cur_indx]):
                cur_indx_array.append(j + 1)
                cur_indx = j + 1
                sign_vec = np.zeros(p)
                sign_vec[np.arange(j + 1 - cur_indx_array[pointer]) + cur_indx_array[pointer]] = \
                    np.sign(self.initial_soln[indices[np.arange(j + 1 - cur_indx_array[pointer]) + cur_indx_array[pointer]]])
                signs_cluster.append(sign_vec)
                pointer = pointer + 1
                if sorted_soln[j + 1] == 0:
                    break

        signs_cluster = np.asarray(signs_cluster).T
        if signs_cluster.size == 0:
            return active_signs

        else:
            X_clustered = X[:, indices].dot(signs_cluster)
            _opt_linear_term = X.T.dot(X_clustered)
            self.opt_transform = (_opt_linear_term, self.initial_subgrad)

            cov, prec = self.randomizer.cov_prec
            opt_linear, opt_offset = self.opt_transform

            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            logdens_linear = cond_cov.dot(opt_linear.T) * prec
            cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

            logdens_transform = (logdens_linear, opt_offset)

            def log_density(logdens_linear, offset, cond_prec, score, opt):
                if score.ndim == 1:
                    mean_term = logdens_linear.dot(score.T + offset).T
                else:
                    mean_term = logdens_linear.dot(score.T + offset[:, None]).T
                arg = opt + mean_term
                return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

            log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

            # now make the constraints

            A_scaling_0 = -np.identity(self.num_opt_var)
            A_scaling_1 = -np.identity(self.num_opt_var)[:(self.num_opt_var - 1), :]
            for k in range(A_scaling_1.shape[0]):
                A_scaling_1[k, k + 1] = 1
            A_scaling = np.vstack([A_scaling_0, A_scaling_1])
            b_scaling = np.zeros(2 * self.num_opt_var - 1)

            affine_con = constraints(A_scaling,
                                     b_scaling,
                                     mean=cond_mean,
                                     covariance=cond_cov)

            self.sampler = affine_gaussian_sampler(affine_con,
                                                   self.observed_opt_state,
                                                   self.observed_score_state,
                                                   log_density,
                                                   logdens_transform,
                                                   selection_info=self.selection_variable)
            return active_signs

    # Targets of inference
    # and covariance with score representation
    # are same as LASSO

    @staticmethod
    def gaussian(X,
                 Y,
                 slope_weights,
                 sigma=1.,
                 quadratic=None,
                 ridge_term=0.,
                 randomizer_scale=None):

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        return slope(loglike, np.asarray(slope_weights) / sigma ** 2, ridge_term, randomizer_scale)

# Projection onto selected subgradients of SLOPE

def _projection_onto_selected_subgradients(prox_arg,
                                           weights,
                                           ordering,
                                           cluster_sizes,
                                           active_signs,
                                           last_value_zero=True):
    """
    Compute the projection of a point onto the set of
    subgradients of the SLOPE penalty with a given
    clustering of the solution and signs of the variables.
    This is a projection onto a lower dimensional set. The dimension
    of this set is p -- the dimensions of the `prox_arg` minus
    the number of unique values in `ordered_clustering` + 1 if the
    last value of the solution was zero (i.e. solution was sparse).

    Parameters
    ----------

    prox_arg : np.ndarray(p, np.float)
        Point to project

    weights : np.ndarray(p, np.float)
        Weights of the SLOPE penalty.

    ordering : np.ndarray(p, np.int)
        Order of original argument to SLOPE prox.
        First entry corresponds to largest argument of SLOPE prox.

    cluster_sizes : sequence
        Sizes of clusters, starting with
        largest in absolute value.

    active_signs : np.ndarray(p, np.int)
         Signs of non-zero coefficients.

    last_value_zero : bool
        Is the last solution value equal to 0?
    """

    result = np.zeros_like(prox_arg)

    ordered_clustering = []
    cur_idx = 0
    for cluster_size in cluster_sizes:
        ordered_clustering.append([ordering[j + cur_idx] for j in range(cluster_size)])
        cur_idx += cluster_size

    # Now, run appropriate SLOPE prox on each cluster
    cur_idx = 0
    for i, cluster in enumerate(ordered_clustering):
        prox_subarg = np.array([prox_arg[j] for j in cluster])

        # If the value of the soln to the prox was non-zero
        # then we solve a SLOPE of size 1 smaller than the cluster

        # If the cluster size is 1, the value is just
        # the corresponding signed weight

        if i < len(ordered_clustering) - 1 or not last_value_zero:
            if len(cluster) == 1:
                result[cluster[0]] = weights[cur_idx] * active_signs[cluster[0]]
            else:
                indices = [j + cur_idx for j in range(len(cluster))]
                cluster_weights = weights[indices]

                ir = IsotonicRegression()
                _ir_result = ir.fit_transform(np.arange(len(cluster)), cluster_weights[::-1])[::-1]
                result[indices] = -np.multiply(active_signs[indices], _ir_result/2.)

        else:
            indices = np.array([j + cur_idx for j in range(len(cluster))])
            cluster_weights = weights[indices]

            slope_prox = _basic_proximal_map(prox_subarg, cluster_weights)
            result[indices] = prox_subarg - slope_prox

        cur_idx += len(cluster)

    return result
