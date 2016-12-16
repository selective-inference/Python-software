import numpy as np
import regreg.api as rr

from .query import query
from .randomization import split

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

    def solve(self, scaling=1, solve_args={'min_its':20, 'tol':1.e-10}):

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

        self._overall = active + unpenalized
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

        self._setup = True
        self._marginalize_subgradient=False
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.p = loss.shape[0]

        self.Q = ((_hessian + epsilon * np.identity(p))[:,active])[active,:]
        self.Qinv = np.linalg.inv(self.Q)
        self.form_VQLambda()


    def form_VQLambda(self):
        nactive_groups = len(self.active_directions_list)
        #print(self.active_directions_list)
        nactive_vars = np.sum([self.active_directions_list[i].shape[0] for i in range(nactive_groups)])
        V = np.zeros((nactive_vars, nactive_vars-nactive_groups))
        #U = np.zeros((nvariables, ngroups))
        Lambda = np.zeros((nactive_vars,nactive_vars))
        temp_row, temp_col = 0, 0
        for g in range(len(self.active_directions_list)):
            size_curr_group = self.active_directions_list[g].shape[0]
            #U[temp_row:(temp_row+size_curr_group),g] = self._active_directions[g]
            Lambda[temp_row:(temp_row+size_curr_group),temp_row:(temp_row+size_curr_group)] \
                = self.active_penalty[g]*np.identity(size_curr_group)
            import scipy
            from scipy import linalg, matrix
            def null(A, eps=1e-12):
                u, s, vh = scipy.linalg.svd(A)
                padding = max(0, np.shape(A)[1] - np.shape(s)[0])
                null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
                null_space = scipy.compress(null_mask, vh, axis=0)
                return scipy.transpose(null_space)

            V_g = null(matrix(self.active_directions_list[g]))
            V[temp_row:(temp_row+V_g.shape[0]), temp_col:(temp_col+V_g.shape[1])] = V_g
            temp_row += V_g.shape[0]
            temp_col += V_g.shape[1]
        self.VQLambda = np.dot(np.dot(V.T,self.Qinv), Lambda.dot(V))

        return self.VQLambda


    def derivative_logdet_jacobian(self, scalings):
        nactive_groups = len(self.active_directions_list)
        nactive_vars = np.sum([self.active_directions_list[i].shape[0] for i in range(nactive_groups)])
        from scipy.linalg import block_diag
        matrix_list = [scalings[i]*np.identity(self.active_directions_list[i].shape[0]-1) for i in range(scalings.shape[0])]
        Gamma_minus = block_diag(*matrix_list)
        jacobian_inv = np.linalg.inv(Gamma_minus+self.VQLambda)

        group_sizes = [self._active_directions[i].shape[0] for i in range(nactive_groups)]
        group_sizes_cumsum = np.concatenate(([0], np.array(group_sizes).cumsum()))

        jacobian_inv_blocks = [jacobian_inv[group_sizes_cumsum[i]:group_sizes_cumsum[i+1],group_sizes_cumsum[i]:group_sizes_cumsum[i+1]]
                                for i in range(nactive_groups)]

        der = np.zeros(self.observed_opt_state.shape[0])
        der[self.scaling_slice] = np.array([np.matrix.trace(jacobian_inv_blocks[i]) for i in range(scalings.shape[0])])
        return der




    def setup_sampler(self, scaling=1, solve_args={'min_its':20, 'tol':1.e-10}):
        pass

    def projection(self, opt_state):
        """
        Full projection for Langevin.

        The state here will be only the state of the optimization variables.
        """

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')


        if ('subgradient' not in self.selection_variable and 
            'scaling' not in self.selection_variable): # have not conditioned on any thing else
            new_state = opt_state.copy() # not really necessary to copy
            new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
            new_state[self.subgrad_slice] = self.group_lasso_dual.bound_prox(opt_state[self.subgrad_slice])
        elif ('subgradient' not in self.selection_variable and
              'scaling' in self.selection_variable): # conditioned on the initial scalings
                                                     # only the subgradient in opt_state
            new_state = self.group_lasso_dual.bound_prox(opt_state)
        elif ('subgradient' in self.selection_variable and
              'scaling' not in self.selection_variable): # conditioned on the subgradient
                                                         # only the scaling in opt_state
            new_state = np.maximum(opt_state, 0)
        else:
            new_state = opt_state
        return new_state

    # optional things to condition on

    def decompose_subgradient(self, conditioning_groups, marginalizing_groups=None):
        """
        Maybe we should allow subgradients of only some variables...
        """
        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        if marginalizing_groups is not None:
            self._marginalize_subgradient=True

        #idx = 0
        groups = np.unique(self.penalty.groups)
        condition_inactive_groups = np.zeros_like(groups, dtype=bool)
        condition_inactive_variables = np.zeros_like(self._inactive, dtype=bool)
        moving_inactive_groups = np.zeros_like(groups, dtype=bool)
        moving_inactive_variables = np.zeros_like(self._inactive, dtype=bool)
        self._inactive_groups = ~(self._active_groups+self._unpenalized)

        inactive_marginal_groups = np.zeros_like(self._inactive, dtype=bool)
        limits_marginal_groups = np.zeros_like(self._inactive)

        for i, g in enumerate(groups):
            if (self._inactive_groups[i]) and conditioning_groups[i]:
                group = self.penalty.groups == g
                condition_inactive_groups[i] = True
                condition_inactive_variables[group] = True
            elif (self._inactive_groups[i]) and (~conditioning_groups[i]) and (~marginalizing_groups[i]):
                group = self.penalty.groups == g
                moving_inactive_groups[i] = True
                moving_inactive_variables[group] = True
            if (self._inactive_groups[i]) and marginalizing_groups[i]:
                group = self.penalty.groups == g
                inactive_marginal_groups[i] = True
                limits_marginal_groups[i] = self.penalty.weights[g]

        self.inactive_marginal_groups = inactive_marginal_groups
        self.limits_marginal_groups = limits_marginal_groups
        #if self.inactive_marginal_groups.sum()==0:
        #    self._marginalize_subgradient=False
                #_opt_affine_term[group] = active_directions[:, idx][group] * penalty.weights[g]
                #idx += 1
        #self.condition_inactive_groups = condition_inactive_groups
        #print("active groups", self._active_groups)
        #print("condtioning", condition_inactive_groups)
        #print("marginalize", self.inactive_marginal_groups)
        opt_linear, opt_offset = self.opt_transform

        new_linear = np.zeros((opt_linear.shape[0], self._active_groups.sum()+self._unpenalized_groups.sum()+moving_inactive_variables.sum()))
        new_linear[:,self.scaling_slice] = opt_linear[:, self.scaling_slice]
        new_linear[:, self.unpenalized_slice] = opt_linear[:, self.unpenalized_slice]

        inactive_moving_idx = np.nonzero(moving_inactive_variables)[0]
        subgrad_idx = range(self._active_groups.sum() + self._unpenalized.sum(),
                            self._active_groups.sum() + self._unpenalized.sum()+moving_inactive_variables.sum())
        subgrad_slice = slice(self._active_groups.sum() + self._unpenalized.sum(),
                              self._active_groups.sum() + self._unpenalized.sum()+moving_inactive_variables.sum())
        for _i, _s in zip(inactive_moving_idx, subgrad_idx):
            new_linear[_i, _s] = 1.

        observed_opt_state = self.observed_opt_state[:(self._active_groups.sum()+self._unpenalized_groups.sum()+moving_inactive_variables.sum())]
        observed_opt_state[subgrad_slice] = self.initial_subgrad[moving_inactive_variables]

        self.observed_opt_state = observed_opt_state

        condition_linear = np.zeros((opt_linear.shape[0], self._active_groups.sum()+self._unpenalized_groups.sum()+condition_inactive_variables.sum()))
        inactive_condition_idx = np.nonzero(condition_inactive_variables)[0]
        subgrad_condition_idx = range(self._active_groups.sum() + self._unpenalized.sum(),
                            self._active_groups.sum() + self._unpenalized.sum() + condition_inactive_variables.sum())
        subgrad_condition_slice = slice(self._active_groups.sum() + self._unpenalized.sum(),
                              self._active_groups.sum() + self._unpenalized.sum() + condition_inactive_variables.sum())
        for _i, _s in zip(inactive_condition_idx, subgrad_condition_idx):
            condition_linear[_i, _s] = 1.

        new_offset = condition_linear[:,subgrad_condition_slice].dot(self.initial_subgrad[condition_inactive_variables]) + opt_offset


        self.opt_transform = (new_linear, new_offset)
        # for group LASSO this should not induce a bigger jacobian as
        # the subgradients are in the interior of a ball
        self.selection_variable['subgradient'] = self.observed_opt_state[self.subgrad_slice]

        # reset variables
        #self.observed_opt_state = np.concatenate((self.observed_opt_state[self.scaling_slice], subgrad_observed[~condition_inactive_variables]), 0)
        #self.scaling_slice = slice(None, None, None)
        #self.subgrad_slice = np.zeros(new_linear.shape[1], np.bool)
        self.num_opt_var = new_linear.shape[1]


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


    def construct_weights(self, full_state):
        """
            marginalizing over the sub-gradient
        """

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        p = self.p
        weights = np.zeros(p)

        if self.inactive_marginal_groups.sum()>0:
            full_state_plus = full_state+np.multiply(self.limits_marginal_groups, np.array(self.inactive_marginal_groups, np.float))
            full_state_minus = full_state-np.multiply(self.limits_marginal_groups, np.array(self.inactive_marginal_groups, np.float))

        #def fraction(upper, lower):
        #    return (self.randomization._pdf(upper) - self.randomization._pdf(lower)) \
        #           / (self.randomization._cdf(upper) - self.randomization._cdf(lower))

        def fraction(full_state_plus, full_state_minus, inactive_marginal_groups):
            return (np.divide(self.randomization._pdf(full_state_plus) - self.randomization._pdf(full_state_minus),
                   self.randomization._cdf(full_state_plus) - self.randomization._cdf(full_state_minus)))[inactive_marginal_groups]
        #for i in range(p):
        #    if self.inactive_marginal_groups[i]:
        #        weights[i] = fraction(full_state_plus[i], full_state_minus[i])
        #    else:
        #        weights[i] = self.randomization._derivative_log_density(full_state[i])

        if self.inactive_marginal_groups.sum()>0:
            weights[self.inactive_marginal_groups] = fraction(full_state_plus, full_state_minus, self.inactive_marginal_groups)
        weights[~self.inactive_marginal_groups] = self.randomization._derivative_log_density(full_state)[~self.inactive_marginal_groups]

        return -weights


def restricted_Mest(Mest_loss, active, solve_args={'min_its':50, 'tol':1.e-10}):

    X, Y = Mest_loss.data

    if Mest_loss._is_transform:
        raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
    X_restricted = X[:,active]
    loss_restricted = rr.affine_smooth(Mest_loss.saturated_loss, X_restricted)
    beta_E = loss_restricted.solve(**solve_args)
    
    return beta_E

class M_estimator_split(M_estimator):

    def __init__(self, loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10}):
        total_size = loss.saturated_loss.shape[0]
        self.randomization = split(loss.shape, subsample_size, total_size)
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

        bootstrap_score = pairs_bootstrap_score(self.loss,
                                                self._overall,
                                                beta_active=self._beta_full[self._overall],
                                                solve_args=solve_args)

        # find unpenalized MLE on subsample

        newq, oldq = rr.identity_quadratic(0, 0, 0, 0), self.randomized_loss.quadratic
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
