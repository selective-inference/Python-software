import numpy as np
import regreg.api as rr
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov
from selection.randomized.query import query
from selection.randomized.randomization import split
import functools

def pairs_bootstrap_glm(glm_loss,
                        active,
                        beta_full=None,
                        inactive=None,
                        scaling=1.,
                        solve_args={'min_its':50, 'tol':1.e-10}):
    """
    pairs bootstrap of (beta_hat_active, -grad_inactive(beta_hat_active))
    """
    X, Y = glm_loss.data

    if beta_full is None:
        beta_active = restricted_Mest(glm_loss, active, solve_args=solve_args)
        beta_full = np.zeros(glm_loss.shape)
        beta_full[active] = beta_active
    else:
        beta_active = beta_full[active]

    X_active = X[:,active]

    nactive = active.sum()
    ntotal = nactive

    if inactive is not None:
        X_inactive = X[:,inactive]
        ntotal += inactive.sum()

    _bootW = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))
    _bootQ = X_active.T.dot(_bootW.dot(X_active))
    _bootQinv = np.linalg.inv(_bootQ)
    if inactive is not None:
        _bootC = X_inactive.T.dot(_bootW.dot(X_active))
        _bootI = _bootC.dot(_bootQinv)
    else:
        _bootI = None

    nactive = active.sum()
    if inactive is not None:
        X_full = np.hstack([X_active,X_inactive])
        beta_overall = np.zeros(X_full.shape[1])
        beta_overall[:nactive] = beta_active
    else:
        X_full = X_active
        beta_overall = beta_active

    _boot_mu = lambda X_full, beta_overall: glm_loss.saturated_loss.mean_function(X_full.dot(beta_overall))

    if ntotal > nactive:
        observed = np.hstack([beta_active, -glm_loss.smooth_objective(beta_full, 'grad')[inactive]])
    else:
        observed = beta_active

    # scaling is a lipschitz constant for a gradient squared
    _sqrt_scaling = np.sqrt(scaling)

    def _boot_score(X_full, Y, ntotal, _bootQinv, _bootI, nactive, _sqrt_scaling, beta_overall, indices):
        X_star = X_full[indices]
        Y_star = Y[indices]
        score = X_star.T.dot(Y_star - _boot_mu(X_star, beta_overall))
        result = np.zeros(ntotal)
        result[:nactive] = _bootQinv.dot(score[:nactive])
        if ntotal > nactive:
            result[nactive:] = score[nactive:] - _bootI.dot(score[:nactive])
        result[:nactive] *= _sqrt_scaling
        result[nactive:] /= _sqrt_scaling
        return result

    observed[:nactive] *= _sqrt_scaling
    observed[nactive:] /= _sqrt_scaling

    return functools.partial(_boot_score, X_full, Y, ntotal, _bootQinv, _bootI, nactive, _sqrt_scaling, beta_overall), observed

def pairs_bootstrap_score(glm_loss,
                          active,
                          beta_active=None,
                          solve_args={'min_its':50, 'tol':1.e-10}):
    """
    pairs bootstrap of (beta_hat_active, -grad_inactive(beta_hat_active))
    """
    X, Y = glm_loss.data

    if beta_active is None:
        beta_active = restricted_Mest(glm_loss, active, solve_args=solve_args)
    X_active = X[:,active]

    _bootW = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))

    _boot_mu = lambda X_active, beta_active: glm_loss.saturated_loss.mean_function(X_active.dot(beta_active))

    def _boot_score(X, Y, active, beta_active, indices):
        X_star = X[indices]
        Y_star = Y[indices]
        score = -X_star.T.dot(Y_star - _boot_mu(X_star[:,active], beta_active))
        return score

    return functools.partial(_boot_score, X, Y, active, beta_active)

def set_alpha_matrix(glm_loss,
                     active,
                     beta_full=None,
                     inactive=None,
                     scaling=1.,
                     solve_args={'min_its': 50, 'tol': 1.e-10}):

    X, Y = glm_loss.data

    if beta_full is None:
        beta_active = restricted_Mest(glm_loss, active, solve_args=solve_args)
        beta_full = np.zeros(glm_loss.shape)
        beta_full[active] = beta_active
    else:
        beta_active = beta_full[active]

    X_active = X[:,active]

    nactive = active.sum()
    ntotal = nactive

    if inactive is not None:
        X_inactive = X[:,inactive]
        ntotal += inactive.sum()

    _W = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))
    _Q = X_active.T.dot(_W.dot(X_active))
    _Qinv = np.linalg.inv(_Q)
    nactive = active.sum()
    if inactive is not None:
        X_full = np.hstack([X_active, X_inactive])
        beta_overall = np.zeros(X_full.shape[1])
        beta_overall[:nactive] = beta_active
    else:
        X_full = X_active
        beta_overall = beta_active

    obs_residuals = Y - glm_loss.saturated_loss.mean_function(X_full.dot(beta_overall))

    return np.dot(np.dot(_Qinv, X_active.T), np.diag(obs_residuals))

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

        self._overall = active + unpenalized
        self._inactive = ~self._overall
        self._unpenalized = unpenalized
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

        self.num_opt_var = p = loss.shape[0] # shorthand for p

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

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # later, we will modify `score_transform`
        # in `linear_decomposition`

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

    def condition_on_subgradient(self):
        """
        Maybe we should allow subgradients of only some variables...
        """
        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        opt_linear, opt_offset = self.opt_transform

        new_offset = opt_linear[:,self.subgrad_slice].dot(self.observed_opt_state[self.subgrad_slice]) + opt_offset
        new_linear = opt_linear[:,self.scaling_slice]

        self.opt_transform = (new_linear, new_offset)

        # for group LASSO this should not induce a bigger jacobian as
        # the subgradients are in the interior of a ball
        self.selection_variable['subgradient'] = self.observed_opt_state[self.subgrad_slice]

        # reset variables

        self.observed_opt_state = self.observed_opt_state[self.scaling_slice]
        self.scaling_slice = slice(None, None, None)
        self.subgrad_slice = np.zeros(new_linear.shape[1], np.bool)
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

        #from .glm import pairs_bootstrap_score

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

        return bootstrap_score, cov

class M_estimator_approx(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer, estimation):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer
        self.estimation = estimation

    def solve_approx(self):

        self.solve()

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate((_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)

        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform

        self._score_linear_term = np.concatenate((_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        self.feasible_point = np.append(self.observed_score_state, np.abs(self.initial_soln[self._overall]))

        lagrange = self.penalty._weight_array

        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        nactive = self._overall.sum()
        self.nactive = nactive

        self.target_observed = self.observed_score_state[:self.nactive]

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))
            inv_X_active = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
            projection_X_active = X[:,self._overall].dot(np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))).dot(X[:,self._overall].T)
            score_cov[:self.nactive, :self.nactive] = inv_X_active
            score_cov[self.nactive:, self.nactive:] = X[:,~self._overall].T.dot(np.identity(n)- projection_X_active).dot(X[:,~self._overall])

        elif self.estimation == 'bootstrap':
            bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        self.score_cov = score_cov
        self.target_cov = score_cov[:nactive, :nactive]
        self.score_cov_inv = np.linalg.inv(self.score_cov)


        self.B = self._opt_linear_term
        self.A = self._score_linear_term

        self.B_active = self.B[:nactive, :nactive]
        self.B_inactive = self.B[nactive:, :nactive]

        self.A_active = self._score_linear_term[:nactive, :]
        self.A_inactive = self._score_linear_term[nactive:, :]

        self.offset_active = self._opt_affine_term[:nactive]


class M_estimator_approx_carved(M_estimator_split):

    def __init__(self, loss, epsilon, subsample_size, penalty, estimation):

        M_estimator_split.__init__(self,loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10})
        self.estimation = estimation

    def solve_approx(self):

        self.solve()

        self.nactive = self._overall.sum()
        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        self.target_observed = self.observed_score_state[:self.nactive]

        self.feasible_point = np.concatenate([self.observed_score_state, np.fabs(self.observed_opt_state[:self.nactive]),
                                              self.observed_opt_state[self.nactive:]], axis = 0)

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)

        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)

        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        lagrange = self.penalty._weight_array

        #print("True or false", np.all(lagrange[0]-np.fabs(self.feasible_point[p+self.nactive:]))>0)
        #print("True or false", np.all(self.feasible_point[p:][:self.nactive]) > 0)

        self.inactive_lagrange = lagrange[~self._overall]

        self.bootstrap_score, self.randomization_cov = self.setup_sampler()

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))
            inv_X_active = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
            projection_X_active = X[:,self._overall].dot(np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))).dot(X[:,self._overall].T)
            score_cov[:self.nactive, :self.nactive] = inv_X_active
            score_cov[self.nactive:, self.nactive:] = X[:,~self._overall].T.dot(np.identity(n)- projection_X_active).dot(X[:,~self._overall])

        elif self.estimation == 'bootstrap':
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), self.bootstrap_score)

        self.score_cov = score_cov
        self.score_cov_inv = np.linalg.inv(self.score_cov)




class M_estimator_approx_logistic(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer, estimation):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer
        self.estimation = estimation

    def solve_approx(self):

        self.solve()

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate((_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)

        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform

        self._score_linear_term = np.concatenate((_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        self.feasible_point = np.append(self.observed_score_state, np.abs(self.initial_soln[self._overall]))

        lagrange = self.penalty._weight_array

        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        nactive = self._overall.sum()
        self.nactive = nactive

        self.target_observed = self.observed_score_state[:self.nactive]

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))

            vec = np.exp(X[:, self._overall].dot(self.target_observed))
            #vec = np.exp(np.zeros(n))
            pi = np.true_divide(vec, np.power(1. + vec, 2))
            weights = np.diag(pi)
            Q_active = X[:, self._overall].T.dot(weights).dot(X[:, self._overall])
            Q_active_inv = np.linalg.inv(Q_active)

            P_inactive = X[:, ~self._overall].T.dot(np.identity(n)-weights.dot(X[:, self._overall].dot(Q_active_inv)
                                                                               .dot(X[:, self._overall].T)))
            score_cov[:self.nactive, :self.nactive] = Q_active_inv
            score_cov[self.nactive:, self.nactive:] = P_inactive.dot(weights).dot(P_inactive.T)

        elif self.estimation == 'bootstrap':
            bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        self.score_cov = score_cov
        self.target_cov = score_cov[:nactive, :nactive]
        self.score_cov_inv = np.linalg.inv(self.score_cov)


        self.B = self._opt_linear_term
        self.A = self._score_linear_term

        self.B_active = self.B[:nactive, :nactive]
        self.B_inactive = self.B[nactive:, :nactive]

        self.A_active = self._score_linear_term[:nactive, :]
        self.A_inactive = self._score_linear_term[nactive:, :]

        self.offset_active = self._opt_affine_term[:nactive]






