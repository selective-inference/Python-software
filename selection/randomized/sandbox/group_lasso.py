from __future__ import print_function
import functools
from copy import copy

import numpy as np
import scipy
from scipy import matrix

import regreg.api as rr
import regreg.affine as ra

from .query import query, optimization_sampler
from .reconstruction import reconstruct_full_from_internal
from .randomization import split

class group_lasso_view(query):

    def __init__(self, loss, epsilon, penalty, randomization, solve_args={'min_its':50, 'tol':1.e-10}):
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
         self.randomization,
         self.solve_args) = (loss,
                             epsilon,
                             penalty,
                             randomization,
                             solve_args)
         
    # Methods needed for subclassing a query

    def solve(self, scaling=1, solve_args={'min_its':20, 'tol':1.e-10}, nboot=2000):

        self.randomize()

        (loss,
         randomized_loss,
         epsilon,
         penalty,
         randomization,
         solve_args) = (self.loss,
                        self.randomized_loss, 
                        self.epsilon,
                        self.penalty,
                        self.randomization,
                        self.solve_args)

        # initial solution

        problem = rr.simple_problem(randomized_loss, penalty)
        self.initial_soln = problem.solve(**solve_args)

        # find the active groups and their direction vectors
        # as well as unpenalized groups

        groups = np.unique(penalty.groups) 
        active_groups = np.zeros(len(groups), np.bool)
        unpenalized_groups = np.zeros(len(groups), np.bool)

        active_directions = []
        active = np.zeros(loss.shape, np.bool)
        unpenalized = np.zeros(loss.shape, np.bool)

        initial_scalings = []

        active_directions_list = [] ## added for group lasso
        active_penalty = []
        for i, g in enumerate(groups):
            group = penalty.groups == g
            active_groups[i] = (np.linalg.norm(self.initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros(active.shape, np.float)
                z[group] = self.initial_soln[group] / np.linalg.norm(self.initial_soln[group])
                active_directions.append(z)
                active_directions_list.append(z[group]) ## added for group lasso
                active_penalty.append(penalty.weights[g]) ## added
                initial_scalings.append(np.linalg.norm(self.initial_soln[group]))
            if unpenalized_groups[i]:
                unpenalized[group] = True

        self.active_penalty = active_penalty

        # solve the restricted problem

        self._overall = active + unpenalized > 0
        self._inactive = ~self._overall
        self._unpenalized = unpenalized

        self.active_directions_list = active_directions_list ## added for group lasso
        self._active_directions = np.array(active_directions).T
        self._active_groups = np.array(active_groups, np.bool)
        self._unpenalized_groups = np.array(unpenalized_groups, np.bool)

        self.selection_variable = {'groups':self._active_groups, 
                                   'variables':self._overall,
                                   'directions':self._active_directions}

        # initial state for opt variables

        initial_subgrad = -(self.randomized_loss.smooth_objective(self.initial_soln, 'grad') + 
                            self.randomized_loss.quadratic.objective(self.initial_soln, 'grad')) 
                          # the quadratic of a smooth_atom is not included in computing the smooth_objective
        self.initial_subgrad = initial_subgrad
        initial_subgrad = initial_subgrad[self._inactive]
        initial_unpenalized = self.initial_soln[self._unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  initial_subgrad], axis=0)

        # set the _solved bit

        self._solved = True

        # Now setup the pieces for linear decomposition

        (loss,
         epsilon,
         penalty,
         initial_soln,
         overall,
         inactive,
         unpenalized,
         active_groups,
         active_directions) = (self.loss,
                               self.epsilon,
                               self.penalty,
                               self.initial_soln,
                               self._overall,
                               self._inactive,
                               self._unpenalized,
                               self._active_groups,
                               self._active_directions)

        # scaling should be chosen to be Lipschitz constant for gradient of Gaussian part

        # we are implicitly assuming that
        # loss is a pairs model

        self.scaling = scaling
        _sqrt_scaling = np.sqrt(self.scaling)

        _beta_unpenalized = restricted_Mest(loss, overall, solve_args=solve_args)

        beta_full = np.zeros(overall.shape)
        beta_full[overall] = _beta_unpenalized
        #_hessian = loss.hessian(beta_full)
        self._beta_full = beta_full

        # observed state for score in internal coordinates

        self.observed_internal_state = np.hstack([_beta_unpenalized * _sqrt_scaling,
                                                  -loss.smooth_objective(beta_full, 'grad')[inactive] / _sqrt_scaling])

        # form linear part
        self.num_opt_var = self.observed_opt_state.shape[0]
        p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self._active_groups.sum() + unpenalized.sum() + inactive.sum()))
        _score_linear_term = np.zeros((p, p))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        Mest_slice = slice(0, overall.sum())
        X, y = loss.data
        W = self.loss.saturated_loss.hessian(X.dot(beta_full))
        _Mest_hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _Mest_hessian_unpen = np.dot(X.T, X[:, unpenalized] * W[:, None])

        _score_linear_term[:, Mest_slice] = -np.hstack([_Mest_hessian_active, _Mest_hessian_unpen]) / _sqrt_scaling

        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        null_idx = range(overall.sum(), p)
        inactive_idx = np.nonzero(inactive)[0]
        for _i, _n in zip(inactive_idx, null_idx):
            _score_linear_term[_i,_n] = -_sqrt_scaling

        # c_E piece 

        scaling_slice = slice(0, active_groups.sum())
        if len(active_directions)==0:
            _opt_hessian=0
        else:
            _opt_hessian = np.dot(_Mest_hessian, active_directions[overall]) + epsilon * active_directions
        _opt_linear_term[:, scaling_slice] = _opt_hessian / _sqrt_scaling

        self.observed_opt_state[scaling_slice] *= _sqrt_scaling

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        unpenalized_directions = np.identity(p)[:,unpenalized]
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (np.dot(_Mest_hessian, unpenalized_directions[overall])
                                                      + epsilon * unpenalized_directions) / _sqrt_scaling
        self.observed_opt_state[unpenalized_slice] *= _sqrt_scaling

        # subgrad piece

        subgrad_idx = range(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i,_s] = _sqrt_scaling

        self.observed_opt_state[subgrad_idx] /= _sqrt_scaling

        # form affine part

        _opt_affine_term = np.zeros(p)
        idx = 0
        groups = np.unique(penalty.groups) 
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _opt_affine_term[group] = active_directions[:,idx][group] * penalty.weights[g]
                idx += 1

        # two transforms that encode score and optimization
        # variable roles 

        # later, we will modify `score_transform`
        # in `linear_decomposition`

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self.scaling_slice = scaling_slice

        # weights are scaled here because the linear terms scales them by scaling

        new_groups = penalty.groups[inactive]
        new_weights = dict([(g, penalty.weights[g] / _sqrt_scaling) for g in penalty.weights.keys() if g in np.unique(new_groups)])

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice

        self._setup = True
        self._marginalize_subgradient = False
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.ndim = loss.shape[0]

        self.nboot = nboot

    def get_sampler(self):
        # setup the default optimization sampler

        if not hasattr(self, "_sampler"):

            def projection(group_lasso_dual, subgrad_slice, scaling_slice, opt_state):
                """
                Full projection for Langevin.

                The state here will be only the state of the optimization variables.
                """

                new_state = opt_state.copy() # not really necessary to copy
                new_state[scaling_slice] = np.maximum(opt_state[scaling_slice], 0)
                new_state[subgrad_slice] = group_lasso_dual.bound_prox(opt_state[subgrad_slice])
                return new_state

            projection = functools.partial(projection, self.group_lasso_dual, self.subgrad_slice, self.scaling_slice)

            def grad_log_density(query,
                                 opt_linear,
                                 rand_gradient,
                                 internal_state,
                                 opt_state):
                full_state = reconstruct_full_from_internal(query.opt_transform, query.score_transform, internal_state, opt_state)
                return opt_linear.T.dot(rand_gradient(full_state).T)

            grad_log_density = functools.partial(grad_log_density, self, self.opt_transform[0], self.randomization.gradient)

            def log_density(query,
                            opt_linear,
                            rand_log_density,
                            internal_state,
                            opt_state):
                full_state = reconstruct_full_from_internal(query.opt_transform, query.score_transform, internal_state, opt_state)
                return rand_log_density(full_state)

            log_density = functools.partial(log_density, self, self.opt_transform[0], self.randomization.log_density)

            self._sampler = optimization_sampler(self.observed_opt_state,
                                                 self.observed_internal_state.copy(),
                                                 self.score_transform,
                                                 self.opt_transform,
                                                 projection,
                                                 grad_log_density,
                                                 log_density)
        return self._sampler

    sampler = property(get_sampler, query.set_sampler)


    def decompose_subgradient(self, conditioning_groups=None, marginalizing_groups=None):
        """
        ADD DOCSTRING

        conditioning_groups and marginalizing_groups should be disjoint
        """

        groups = np.unique(self.penalty.groups)
        condition_inactive_groups = np.zeros_like(groups, dtype=bool)

        if conditioning_groups is None:
            conditioning_groups = np.zeros_like(groups, dtype=np.bool)

        if marginalizing_groups is None:
            marginalizing_groups = np.zeros_like(groups, dtype=np.bool)

        if np.any(conditioning_groups * marginalizing_groups):
            raise ValueError("cannot simultaneously condition and marginalize over a group's subgradient")

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        condition_inactive_variables = np.zeros_like(self._inactive, dtype=bool)
        moving_inactive_groups = np.zeros_like(groups, dtype=bool)
        moving_inactive_variables = np.zeros_like(self._inactive, dtype=bool)
        _inactive_groups = ~(self._active_groups+self._unpenalized)

        inactive_marginal_groups = np.zeros_like(self._inactive, dtype=bool)
        limits_marginal_groups = np.zeros_like(self._inactive, np.float)

        for i, g in enumerate(groups):
            if (_inactive_groups[i]) and conditioning_groups[i]:
                group = self.penalty.groups == g
                condition_inactive_groups[i] = True
                condition_inactive_variables[group] = True
            elif (_inactive_groups[i]) and (~conditioning_groups[i]) and (~marginalizing_groups[i]):
                group = self.penalty.groups == g
                moving_inactive_groups[i] = True
                moving_inactive_variables[group] = True
            if (_inactive_groups[i]) and marginalizing_groups[i]:
                group = self.penalty.groups == g
                inactive_marginal_groups[i] = True
                limits_marginal_groups[i] = self.penalty.weights[g]

        opt_linear, opt_offset = self.opt_transform

        new_linear = np.zeros((opt_linear.shape[0], (self._active_groups.sum() +
                                                     self._unpenalized_groups.sum() +
                                                     moving_inactive_variables.sum())))
        new_linear[:, self.scaling_slice] = opt_linear[:, self.scaling_slice]
        new_linear[:, self.unpenalized_slice] = opt_linear[:, self.unpenalized_slice]

        inactive_moving_idx = np.nonzero(moving_inactive_variables)[0]
        subgrad_idx = range(self._active_groups.sum() + self._unpenalized.sum(),
                            self._active_groups.sum() + self._unpenalized.sum() +
                            moving_inactive_variables.sum())
        subgrad_slice = subgrad_idx
        for _i, _s in zip(inactive_moving_idx, subgrad_idx):
            new_linear[_i, _s] = 1.

        observed_opt_state = self.observed_opt_state[:(self._active_groups.sum() +
                                                       self._unpenalized_groups.sum() +
                                                       moving_inactive_variables.sum())]
        observed_opt_state[subgrad_idx] = self.initial_subgrad[moving_inactive_variables]

        condition_linear = np.zeros((opt_linear.shape[0], (self._active_groups.sum() +
                                                           self._unpenalized_groups.sum() +
                                                           condition_inactive_variables.sum())))
        inactive_condition_idx = np.nonzero(condition_inactive_variables)[0]
        subgrad_condition_idx = range(self._active_groups.sum() + self._unpenalized.sum(),
                                      self._active_groups.sum() + self._unpenalized.sum() + condition_inactive_variables.sum())

        for _i, _s in zip(inactive_condition_idx, subgrad_condition_idx):
            condition_linear[_i, _s] = 1.

        new_offset = condition_linear[:,subgrad_condition_idx].dot(self.initial_subgrad[condition_inactive_variables]) + opt_offset

        new_opt_transform = (new_linear, new_offset)

        print("limits marginal groups", limits_marginal_groups)
        print("inactive marginal groups", inactive_marginal_groups)

        def _fraction(_cdf, _pdf, full_state_plus, full_state_minus, inactive_marginal_groups):
            return (np.divide(_pdf(full_state_plus) - _pdf(full_state_minus),
                              _cdf(full_state_plus) - _cdf(full_state_minus)))[inactive_marginal_groups]

        def new_grad_log_density(query, 
                                 limits_marginal_groups,
                                 inactive_marginal_groups,
                                 _cdf,
                                 _pdf,
                                 opt_linear,
                                 deriv_log_dens,
                                 internal_state, 
                                 opt_state):

            full_state = reconstruct_full_from_internal(new_opt_transform, query.score_transform, internal_state, opt_state)

            p = query.penalty.shape[0]
            weights = np.zeros(p)

            if inactive_marginal_groups.sum()>0:
                full_state_plus = full_state + np.multiply(limits_marginal_groups, np.array(inactive_marginal_groups, np.float))
                full_state_minus = full_state - np.multiply(limits_marginal_groups, np.array(inactive_marginal_groups, np.float))
                weights[inactive_marginal_groups] = _fraction(_cdf, _pdf, full_state_plus, full_state_minus, inactive_marginal_groups)
            weights[~inactive_marginal_groups] = deriv_log_dens(full_state)[~inactive_marginal_groups]
            return -opt_linear.T.dot(weights)

        new_grad_log_density = functools.partial(new_grad_log_density,
                                                 self,
                                                 limits_marginal_groups,
                                                 inactive_marginal_groups,
                                                 self.randomization._cdf,
                                                 self.randomization._pdf,
                                                 new_opt_transform[0],
                                                 self.randomization._derivative_log_density)

        def new_log_density(query, 
                            limits_marginal_groups,
                            inactive_marginal_groups,
                            _cdf,
                            _pdf,
                            opt_linear,
                            log_dens,
                            internal_state, 
                            opt_state):

            full_state = reconstruct_full_from_internal(new_opt_transform,
                                                        query.score_transform,
                                                        internal_state,
                                                        opt_state)
            full_state = np.atleast_2d(full_state)
            p = query.penalty.shape[0]
            logdens = np.zeros(full_state.shape[0])

            if inactive_marginal_groups.sum()>0:
                full_state_plus = full_state + np.multiply(limits_marginal_groups, np.array(inactive_marginal_groups, np.float))
                full_state_minus = full_state - np.multiply(limits_marginal_groups, np.array(inactive_marginal_groups, np.float))
                logdens += np.sum(np.log(_cdf(full_state_plus) - _cdf(full_state_minus))[:,inactive_marginal_groups], axis=1)

            logdens += log_dens(full_state[:,~inactive_marginal_groups])

            return np.squeeze(logdens) # should this be negative to match the gradient log density?

        new_log_density = functools.partial(new_log_density,
                                            self,
                                            limits_marginal_groups,
                                            inactive_marginal_groups,
                                            self.randomization._cdf,
                                            self.randomization._pdf,
                                            self.opt_transform[0],
                                            self.randomization._log_density)

        new_groups = self.penalty.groups[moving_inactive_groups]
        _sqrt_scaling = np.sqrt(self.scaling)
        new_weights = dict([(g, self.penalty.weights[g] / _sqrt_scaling) for g in self.penalty.weights.keys() if g in np.unique(new_groups)])
        new_group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)

        def new_projection(group_lasso_dual,
                           noverall,
                           opt_state):
            new_state = opt_state.copy()
            new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
            new_state[noverall:] = group_lasso_dual.bound_prox(opt_state[noverall:])
            return new_state

        new_projection = functools.partial(new_projection,
                                           new_group_lasso_dual,
                                           self._overall.sum())
                                           
        new_selection_variable = copy(self.selection_variable)
        new_selection_variable['subgradient'] = self.observed_opt_state[self.subgrad_slice]

        self.sampler = optimization_sampler(observed_opt_state,
                                            self.observed_internal_state.copy(),
                                            self.score_transform,
                                            new_opt_transform,
                                            new_projection,
                                            new_grad_log_density,
                                            new_log_density,
                                            selection_info=(self, new_selection_variable))

    def condition_on_scalings(self):
        """
        Maybe we should allow subgradients of only some variables...
        """
        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        opt_linear, opt_offset = self.opt_transform
        
        new_offset = opt_linear[:,self.scaling_slice].dot(self.observed_opt_state[self.scaling_slice]) + opt_offset
        new_linear = opt_linear[:,self.subgrad_slice]

        self.opt_transform = (new_linear, new_offset)

        # for group LASSO this will induce a bigger jacobian
        self.selection_variable['scalings'] = self.observed_opt_state[self.scaling_slice]

        # reset slices 

        self.observed_opt_state = self.observed_opt_state[self.subgrad_slice]
        self.subgrad_slice = slice(None, None, None)
        self.scaling_slice = np.zeros(new_linear.shape[1], np.bool)
        self.num_opt_var = new_linear.shape[1]

#     def grad_log_density(self, internal_state, opt_state):
#         """
#             marginalizing over the sub-gradient

#             full_state is 
#             density should be expressed in terms of opt_state coordinates
#         """

#         if not self._setup:
#             raise ValueError('setup_sampler should be called before using this function')

#         if self._marginalize_subgradient:

#             full_state = reconstruct_full_from_internal(self, internal_state, opt_state)

#             p = self.penalty.shape[0]
#             weights = np.zeros(p)

#             if self.inactive_marginal_groups.sum()>0:
#                 full_state_plus = full_state + np.multiply(self.limits_marginal_groups, np.array(self.inactive_marginal_groups, np.float))
#                 full_state_minus = full_state - np.multiply(self.limits_marginal_groups, np.array(self.inactive_marginal_groups, np.float))


#             def fraction(full_state_plus, full_state_minus, inactive_marginal_groups):
#                 return (np.divide(self.randomization._pdf(full_state_plus) - self.randomization._pdf(full_state_minus),
#                        self.randomization._cdf(full_state_plus) - self.randomization._cdf(full_state_minus)))[inactive_marginal_groups]

#             if self.inactive_marginal_groups.sum() > 0:
#                 weights[self.inactive_marginal_groups] = fraction(full_state_plus, full_state_minus, self.inactive_marginal_groups)
#             weights[~self.inactive_marginal_groups] = self.randomization._derivative_log_density(full_state)[~self.inactive_marginal_groups]

#             opt_linear = self.opt_transform[0]
#             return -opt_linear.T.dot(weights)
#         else:
#             return query.grad_log_density(self, internal_state, opt_state)

def restricted_Mest(Mest_loss, active, solve_args={'min_its':50, 'tol':1.e-10}):
    """
    Fit a restricted model using only columns `active`.

    Parameters
    ----------

    Mest_loss : objective function
        A GLM loss.

    active : ndarray
        Which columns to use.

    solve_args : dict
        Passed to `solve`.

    Returns
    -------

    soln : ndarray
        Solution to restricted problem.

    """
    X, Y = Mest_loss.data

    if not Mest_loss._is_transform and hasattr(Mest_loss, 'saturated_loss'): # M_est is a glm
        X_restricted = X[:,active]
        loss_restricted = rr.affine_smooth(Mest_loss.saturated_loss, X_restricted)
    else:
        I_restricted = ra.selector(active, ra.astransform(X).input_shape[0], ra.identity((active.sum(),)))
        loss_restricted = rr.affine_smooth(Mest_loss, I_restricted.T)
    beta_E = loss_restricted.solve(**solve_args)
    
    return beta_E

class group_lasso_split(group_lasso_view):

    def __init__(self, loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10}):

        total_size = loss.saturated_loss.shape[0]
        self.randomization = split(loss.shape, subsample_size, total_size)

        group_lasso.__init__(self, loss, epsilon, penalty, self.randomization, solve_args=solve_args)

        total_size = loss.saturated_loss.shape[0]
        if subsample_size > total_size:
            raise ValueError('subsample size must be smaller than total sample size')

        self.total_size, self.subsample_size = total_size, subsample_size
        

class group_lasso_group_lasso(group_lasso_view):

    def __init__(self, loss, epsilon, penalty, randomization, solve_args={'min_its': 50, 'tol': 1.e-10}):

        group_lasso.__init__(self, loss, epsilon, penalty, randomization, solve_args=solve_args)

        self.Q = self._Mest_hessian[self._overall,:] + epsilon * np.identity(self._overall.sum())
        self.Qinv = np.linalg.inv(self.Q)
        self.form_VQLambda()

    def form_VQLambda(self):
        nactive_groups = len(self.active_directions_list)
        nactive_vars = sum([self.active_directions_list[i].shape[0] for i in range(nactive_groups)])
        V = np.zeros((nactive_vars, nactive_vars - nactive_groups))

        Lambda = np.zeros((nactive_vars, nactive_vars))
        temp_row, temp_col = 0, 0
        for g in range(len(self.active_directions_list)):
            size_curr_group = self.active_directions_list[g].shape[0]

            Lambda[temp_row:(temp_row + size_curr_group), temp_row:(temp_row + size_curr_group)] \
                = self.active_penalty[g] * np.identity(size_curr_group)

            def null(A, eps=1e-12):
                u, s, vh = np.linalg.svd(A)
                padding = max(0, np.shape(A)[1] - np.shape(s)[0])
                null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
                null_space = scipy.compress(null_mask, vh, axis=0)
                return scipy.transpose(null_space)

            V_g = null(matrix(self.active_directions_list[g]))
            V[temp_row:(temp_row + V_g.shape[0]), temp_col:(temp_col + V_g.shape[1])] = V_g
            temp_row += V_g.shape[0]
            temp_col += V_g.shape[1]
        self.VQLambda = np.dot(np.dot(V.T, self.Qinv), Lambda.dot(V))

        return self.VQLambda

    def derivative_logdet_jacobian(self, scalings):
        nactive_groups = len(self.active_directions_list)
        nactive_vars = np.sum([self.active_directions_list[i].shape[0] for i in range(nactive_groups)])
        from scipy.linalg import block_diag
        matrix_list = [scalings[i] * np.identity(self.active_directions_list[i].shape[0] - 1) for i in
                       range(scalings.shape[0])]
        Gamma_minus = block_diag(*matrix_list)
        jacobian_inv = np.linalg.inv(Gamma_minus + self.VQLambda)

        group_sizes = [self._active_directions[i].shape[0] for i in range(nactive_groups)]
        group_sizes_cumsum = np.concatenate(([0], np.array(group_sizes).cumsum()))

        jacobian_inv_blocks = [
            jacobian_inv[group_sizes_cumsum[i]:group_sizes_cumsum[i + 1],
            group_sizes_cumsum[i]:group_sizes_cumsum[i + 1]]
            for i in range(nactive_groups)]

        der = np.zeros(self.observed_opt_state.shape[0])
        der[self.scaling_slice] = np.array([np.matrix.trace(jacobian_inv_blocks[i]) for i in range(scalings.shape[0])])
        return der


#### Subclasses of different randomized views

class glm_group_lasso(group_lasso_view):

    def setup_sampler(self, scaling=1., solve_args={'min_its':50, 'tol':1.e-10}):

        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self.selection_variable['variables'],
                                              beta_full=self._beta_full,
                                              inactive=~self.selection_variable['variables'])[0]

        return bootstrap_score

class split_glm_group_lasso(group_lasso_split):

    def setup_sampler(self, scaling=1., solve_args={'min_its': 50, 'tol': 1.e-10}, B=1000):

        # now we need to estimate covariance of
        # loss.grad(\beta_E^*) - 1/pi * randomized_loss.grad(\beta_E^*)

        m, n, p = self.subsample_size, self.total_size, self.loss.shape[0] # shorthand
        
        from .glm import pairs_bootstrap_score # need to correct these imports!!!

        bootstrap_score = pairs_bootstrap_score(self.loss,
                                                self._overall,
                                                beta_active=self._beta_full[self._overall],
                                                solve_args=solve_args)

        # find unpenalized MLE on subsample

        newq, oldq = identity_quadratic(0, 0, 0, 0), self.randomized_loss.quadratic
        self.randomized_loss.quadratic = newq
        beta_active_subsample = restricted_Mest(self.randomized_loss,
                                                self._overall)

        bootstrap_score_split = pairs_bootstrap_score(self.loss,
                                                      self._overall,
                                                      beta_active=beta_active_subsample,
                                                      solve_args=solve_args)
        self.randomized_loss.quadratic = oldq

        inv_frac = n / m
        
        def subsample_diff(m, n, indices):
            subsample = np.random.choice(indices, size=m, replace=False)
            full_score = bootstrap_score(indices) # a sum of n terms
            randomized_score = bootstrap_score_split(subsample) # a sum of m terms
            return full_score - randomized_score * inv_frac

        first_moment = np.zeros(p)
        second_moment = np.zeros((p, p))
        
        _n = np.arange(n)
        for _ in range(B):
            indices = np.random.choice(_n, size=n, replace=True)
            randomized_score = subsample_diff(m, n, indices)
            first_moment += randomized_score
            second_moment += np.multiply.outer(randomized_score, randomized_score)

        first_moment /= B
        second_moment /= B

        cov = second_moment - np.multiply.outer(first_moment,
                                                first_moment)

        self.randomization.set_covariance(cov)

        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self.selection_variable['variables'],
                                              beta_full=self._beta_full,
                                              inactive=~self.selection_variable['variables'])[0]

        return bootstrap_score
