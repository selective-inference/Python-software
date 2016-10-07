import numpy as np
import regreg.api as rr

from .query import query
from .randomization import split, splitJT

class M_estimator(query):

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

    def solve(self):

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

        for i, g in enumerate(groups):
            group = penalty.groups == g
            active_groups[i] = (np.linalg.norm(self.initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros(active.shape, np.float)
                z[group] = self.initial_soln[group] / np.linalg.norm(self.initial_soln[group])
                active_directions.append(z)
                initial_scalings.append(np.linalg.norm(self.initial_soln[group]))
            if unpenalized_groups[i]:
                unpenalized[group] = True

        # solve the restricted problem

        self.overall = active + unpenalized
        self.inactive = ~self.overall
        self.unpenalized = unpenalized
        self.active_directions = np.array(active_directions).T
        self.active_groups = np.array(active_groups, np.bool)
        self.unpenalized_groups = np.array(unpenalized_groups, np.bool)

        self.selection_variable = {'groups':self.active_groups, 
                                   'directions':self.active_directions}

        # initial state for opt variables

        initial_subgrad = -(self.randomized_loss.smooth_objective(self.initial_soln, 'grad') + 
                            self.randomized_loss.quadratic.objective(self.initial_soln, 'grad')) 
                          # the quadratic of a smooth_atom is not included in computing the smooth_objective

        initial_subgrad = initial_subgrad[self.inactive]
        initial_unpenalized = self.initial_soln[self.unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  initial_subgrad], axis=0)

        # set the _solved bit

        self._solved = True

    def setup_sampler(self, scaling=1., solve_args={'min_its':50, 'tol':1.e-10}):

        """
        Should return a bootstrap_score
        """

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
                               self.overall,
                               self.inactive,
                               self.unpenalized,
                               self.active_groups,
                               self.active_directions)

        # scaling should be chosen to be Lipschitz constant for gradient of Gaussian part

        # we are implicitly assuming that
        # loss is a pairs model

        _sqrt_scaling = np.sqrt(scaling)

        _beta_unpenalized = restricted_Mest(loss, overall, solve_args=solve_args)

        beta_full = np.zeros(overall.shape)
        beta_full[overall] = _beta_unpenalized
        _hessian = loss.hessian(beta_full)
        self._beta_full = beta_full

        # observed state for score

        self.observed_score_state = np.hstack([_beta_unpenalized * _sqrt_scaling,
                                               -loss.smooth_objective(beta_full, 'grad')[inactive] / _sqrt_scaling])

        # form linear part

        self.num_opt_var = p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self.active_groups.sum() + unpenalized.sum() + inactive.sum()))
        _score_linear_term = np.zeros((p, p))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        Mest_slice = slice(0, overall.sum())
        _Mest_hessian = _hessian[:,overall]
        _score_linear_term[:,Mest_slice] = -_Mest_hessian / _sqrt_scaling

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
            _opt_hessian = (_hessian + epsilon * np.identity(p)).dot(active_directions)
        _opt_linear_term[:,scaling_slice] = _opt_hessian / _sqrt_scaling

        self.observed_opt_state[scaling_slice] *= _sqrt_scaling

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        unpenalized_directions = np.identity(p)[:,unpenalized]
        if unpenalized.sum():
            _opt_linear_term[:,unpenalized_slice] = (_hessian + epsilon * np.identity(p)).dot(unpenalized_directions) / _sqrt_scaling

        self.observed_opt_state[unpenalized_slice] *= _sqrt_scaling

        # subgrad piece

        subgrad_idx = range(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i,_s] = _sqrt_scaling

        self.observed_opt_state[subgrad_slice] /= _sqrt_scaling

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

    def projection(self, opt_state):
        """
        Full projection for Langevin.

        The state here will be only the state of the optimization variables.
        """

        if not hasattr(self, "scaling_slice"):
            raise ValueError('setup_sampler should be called before using this function')

        new_state = opt_state.copy() # not really necessary to copy
        new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
        new_state[self.subgrad_slice] = self.group_lasso_dual.bound_prox(opt_state[self.subgrad_slice])

        return new_state

def restricted_Mest(Mest_loss, active, solve_args={'min_its':50, 'tol':1.e-10}):

    X, Y = Mest_loss.data

    if Mest_loss._is_transform:
        raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
    X_restricted = X[:,active]
    loss_restricted = rr.affine_smooth(Mest_loss.saturated_loss, X_restricted)
    beta_E = loss_restricted.solve(**solve_args)
    
    return beta_E


class split_M_estimator(M_estimator):

    def __init__(self,loss, epsilon, penalty, randomization, solve_args={'min_its':50, 'tol':1.e-10}):
        M_estimator.__init__(self,loss, epsilon, penalty, randomization, solve_args=solve_args)

    def randomize(self):

        if not self._randomized:
            # self._randomZ = self.randomization.sample()
            self._randomZ = np.dot(self.loss.X1.T, self.loss.y1) - self.loss.fraction*np.dot(self.loss.X.T, self.loss.y)
            self._random_term = rr.identity_quadratic(self.epsilon, 0, -self._randomZ, 0)

        # set the _randomized bit

        self._randomized = True

    def setup_sampler(self, scaling=1., solve_args={'min_its': 50, 'tol': 1.e-10}):

        """
        Should return a bootstrap_score
        """

        (loss,
         epsilon,
         penalty,
         randomization,
         initial_soln,
         overall,
         inactive,
         unpenalized,
         active_groups,
         active_directions) = (self.loss,
                               self.epsilon,
                               self.penalty,
                               self.randomization,
                               self.initial_soln,
                               self.overall,
                               self.inactive,
                               self.unpenalized,
                               self.active_groups,
                               self.active_directions)

        # scaling should be chosen to be Lipschitz constant for gradient of Gaussian part

        # we are implicitly assuming that
        # loss is a pairs model

        self.randomization = split(loss,overall)

        _sqrt_scaling = np.sqrt(scaling)

        _beta_unpenalized1 = restricted_Mest(loss.sub_loss, overall, solve_args=solve_args)

        beta_full1 = np.zeros(overall.shape)
        beta_full1[overall] = _beta_unpenalized1
        _hessian = loss.sub_loss.hessian(beta_full1)
        self._beta_full1 = beta_full1

        _beta_unpenalized = restricted_Mest(loss.full_loss, overall, solve_args=solve_args)
        beta_full = np.zeros(overall.shape)
        beta_full[overall] = _beta_unpenalized
        #_hessian = loss.sub_loss.hessian(beta_full)
        self._beta_full = beta_full

        # observed state for score

        #self.observed_score_state = np.hstack([_beta_unpenalized * _sqrt_scaling,
        #                                       -loss.smooth_objective(beta_full, 'grad')[inactive] / _sqrt_scaling])

        #self.observed_score_state = np.dot(loss.X1.T,loss.sub_loss.saturated_loss.smooth_objective(loss.X1.dot(beta_full1), 'grad') + loss.y1)

        #self.observed_score_state += -np.dot(loss.X.T, loss.y)*loss.fraction-np.dot(_hessian1, beta_full1)

        self.observed_score_state = loss.smooth_objective(initial_soln, 'grad')

        #print loss.smooth_objective(initial_soln, 'grad')
        # form linear part

        self.num_opt_var = p = loss.shape[0]  # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self.active_groups.sum() + unpenalized.sum() + inactive.sum()))
        _score_linear_term = np.identity(p)

        #_score_linear_term = np.zeros((p, p))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        #Mest_slice = slice(0, overall.sum())
        #_Mest_hessian = _hessian[:, overall]
        #_score_linear_term[:, Mest_slice] = -_Mest_hessian / _sqrt_scaling



        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        #null_idx = range(overall.sum(), p)
        inactive_idx = np.nonzero(inactive)[0]
        #for _i, _n in zip(inactive_idx, null_idx):
        #    _score_linear_term[_i, _n] = -_sqrt_scaling

        # c_E piece

        scaling_slice = slice(0, active_groups.sum())
        if len(active_directions) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = (_hessian + epsilon * np.identity(p)).dot(active_directions)
        _opt_linear_term[:, scaling_slice] = _opt_hessian / _sqrt_scaling

        self.observed_opt_state[scaling_slice] *= _sqrt_scaling

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        unpenalized_directions = np.identity(p)[:, unpenalized]
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (_hessian + epsilon * np.identity(p)).dot(
                unpenalized_directions) / _sqrt_scaling

        self.observed_opt_state[unpenalized_slice] *= _sqrt_scaling

        # subgrad piece

        subgrad_idx = range(active_groups.sum() + unpenalized.sum(),
                            active_groups.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(),
                              active_groups.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i, _s] = _sqrt_scaling

        self.observed_opt_state[subgrad_slice] /= _sqrt_scaling

        # form affine part

        _opt_affine_term = np.zeros(p)
        idx = 0
        groups = np.unique(penalty.groups)
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _opt_affine_term[group] = active_directions[:, idx][group] * penalty.weights[g]
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
        new_weights = dict(
            [(g, penalty.weights[g] / _sqrt_scaling) for g in penalty.weights.keys() if g in np.unique(new_groups)])

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice


class M_estimator_splitJT(M_estimator):

    def __init__(self, loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10}):
        total_size = loss.saturated_loss.shape[0]
        self.randomization = splitJT(loss.shape, subsample_size, total_size)
        M_estimator.__init__(self,loss, epsilon, penalty, self.randomization, solve_args=solve_args)

        total_size = loss.saturated_loss.shape[0]
        if subsample_size > total_size:
            raise ValueError('subsample size must be smaller than total sample size')

        self.total_size, self.subsample_size = total_size, subsample_size

    def setup_sampler(self, scaling=1., solve_args={'min_its': 50, 'tol': 1.e-10}, B=2000):

        M_estimator.setup_sampler(self, 
                                  scaling=scaling,
                                  solve_args=solve_args)
        
        # now we need to estimate covariance of
        # loss.grad(\beta_E^*) - 1/pi * randomized_loss.grad(\beta_E^*)

        m, n, p = self.subsample_size, self.total_size, self.loss.shape[0] # shorthand
        
        from .glm import pairs_bootstrap_score # need to correct these imports!!!

        print(self._beta_full)
        bootstrap_score = pairs_bootstrap_score(self.loss,
                                                self.overall,
                                                beta_active=self._beta_full[self.overall],
                                                solve_args=solve_args)

        inv_frac = n / m
        
        def subsample_diff(m, n, indices):
            subsample = np.random.choice(indices, size=m, replace=False)
            full_score = bootstrap_score(indices) # a sum of n terms
            randomized_score = bootstrap_score(subsample) # a sum of m terms
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
