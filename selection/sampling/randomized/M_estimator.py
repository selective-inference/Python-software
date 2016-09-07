import numpy as np
from regreg.smooth.glm import glm as regreg_glm, logistic_loglike
import regreg.api as rr

class M_estimator(object):

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

        (self.loss,
         self.epsilon,
         self.penalty,
         self.randomization,
         self.solve_args) = (loss,
                             epsilon,
                             penalty,
                             randomization,
                             solve_args)
         
    def solve(self):

        (loss,
         epsilon,
         penalty,
         randomization,
         solve_args) = (self.loss,
                        self.epsilon,
                        self.penalty,
                        self.randomization,
                        self.solve_args)

        # initial solution

        problem = rr.simple_problem(loss, penalty)
        self._randomZ = self.randomization.sample()
        self._random_term = rr.identity_quadratic(epsilon, 0, -self._randomZ, 0)
        self.initial_soln = problem.solve(self._random_term, **solve_args)

    def setup_sampler(self, solve_args={'min_its':50, 'tol':1.e-10}):

        (loss,
         epsilon,
         penalty,
         randomization,
         initial_soln) = (self.loss,
                          self.epsilon,
                          self.penalty,
                          self.randomization,
                          self.initial_soln)

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
            active_groups[i] = (np.linalg.norm(initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros(active.shape, np.float)
                z[group] = initial_soln[group] / np.linalg.norm(initial_soln[group])
                active_directions.append(z)
                initial_scalings.append(np.linalg.norm(initial_soln[group]))
            if unpenalized_groups[i]:
                unpenalized[group] = True

        # solve the restricted problem

        overall = active + unpenalized
        inactive = ~overall

        # initial state for opt variables

        initial_subgrad = -(self.loss.smooth_objective(self.initial_soln, 'grad') + self._random_term.objective(self.initial_soln, 'grad'))
        initial_subgrad = initial_subgrad[inactive]
        initial_unpenalized = self.initial_soln[unpenalized]
        self._initial_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  initial_subgrad], axis=0)

        active_directions = np.array(active_directions).T

        # we are implicitly assuming that
        # loss is a pairs model

        _beta_unpenalized = restricted_glm(loss, overall, solve_args=solve_args)

        beta_full = np.zeros(active.shape)
        beta_full[active] = _beta_unpenalized
        _hessian = loss.hessian(beta_full)
        self._beta_full = beta_full

        # initial state for score

        self._initial_score_state = np.hstack([_beta_unpenalized,
                                               loss.smooth_objective(beta_full, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, p))
        _score_linear_term = np.zeros((p, active_groups.sum() + unpenalized.sum() + inactive.sum()))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        Mest_slice = slice(0, overall.sum())
        _Mest_hessian = _hessian[:,overall]
        _score_linear_term[:,Mest_slice] = -_Mest_hessian

        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        null_slice = slice(overall.sum(), p)
        _score_linear_term[inactive][:,null_slice] = -np.identity(inactive.sum())

        # c_E piece 

        scaling_slice = slice(0, active_groups.sum())
        _opt_hessian = (_hessian + epsilon * np.identity(p)).dot(active_directions)
        _opt_linear_term[:,scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        if unpenalized.sum():
            _opt_linear_term[:,unpenalized_slice] = _hessian[:,unpenalized] + epsilon * np.identity(unpenalized.sum())

        # subgrad piece
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        _opt_linear_term[inactive][:,subgrad_slice] = np.identity(inactive.sum())

        # form affine part

        _opt_affine_term = np.zeros(p)
        idx = 0
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _opt_affine_term[group] = active_directions[:,idx][group] * penalty.weights[g]
                idx += 1

        # two transforms that encode score and optimization
        # variable roles 

        # later, conditioning will modify `score_transform`

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self.scaling_slice = scaling_slice

        new_groups = penalty.groups[inactive]
        new_weights = dict([(g,penalty.weights[g]) for g in penalty.weights.keys() if g in np.unique(new_groups)])

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice

        # store active sets, etc.

        (self.overall,
         self.active,
         self.unpenalized,
         self.inactive) = (overall,
                           active,
                           unpenalized,
                           inactive)

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

    def gradient(self, data_state, data_transform, opt_state):
        """
        Randomization derivative at full state.
        """

        if not hasattr(self, "opt_transform"):
            raise ValueError('setup_sampler should be called before using this function')

        # omega
        opt_linear, opt_offset = self.opt_transform
        data_linear, data_offset = data_transform
        data_piece = data_linear.dot(data_state) + data_offset
        opt_piece = opt_linear.dot(opt_state) + opt_offset
        full_state = (data_piece + opt_piece)
        randomization_derivative = self.randomization.gradient(full_state)
        data_grad = data_linear.T.dot(randomization_derivative)
        opt_grad = opt_linear.T.dot(randomization_derivative)
        return data_grad, opt_grad


    def condition(self, target_score_cov, target_cov, target_observed_state):
        """
        condition the score on the target,
        return a new score_transform
        that is composition of `self.score_transform`
        with the affine map from conditioning
        """

        target_score_cov = np.atleast_2d(target_score_cov) 
        target_cov = np.atleast_2d(target_cov) 
        target_observed_state = np.atleast_1d(target_observed_state)

        linear_part = target_score_cov.T.dot(np.linalg.pinv(target_cov))
        offset = self._initial_score_state - linear_part.dot(target_observed_state)

        # now compute the composition of this map with
        # self.score_transform

        score_linear, score_offset = self.score_transform
        composition_linear_part = score_linear.dot(linear_part)
        composition_offset = score_linear.dot(offset) + score_offset

        return (composition_linear_part, composition_offset)

def restricted_glm(glm_loss, active, solve_args={'min_its':50, 'tol':1.e-10}):

    X, Y = glm_loss.data

    if glm_loss._is_transform:
        raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
    X_restricted = X[:,active]
    loss_restricted = rr.affine_smooth(glm_loss.saturated_loss, X_restricted)
    beta_E = loss_restricted.solve(**solve_args)
    
    return beta_E

def pairs_bootstrap_glm(glm_loss, active, beta_full=None, inactive=None, solve_args={'min_its':50, 'tol':1.e-10}):

    X, Y = glm_loss.data

    if beta_full is None:
        beta_active = restricted_glm(glm_loss, active, solve_args=solve_args)
    else:
        beta_active = beta_full[active]


    X_active = X[:,active]
    _boot_mu = lambda X: glm_loss.saturated_loss.smooth_objective(X_active.dot(beta_active), 'grad') + Y

    nactive = active.sum()
    ntotal = nactive

    if inactive is not None:
        X_inactive = X[:,inactive]
        ntotal += inactive.sum()

    nactive = active.sum()
    _bootW = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))
    _bootQ = X_active.T.dot(_bootW.dot(X_active))
    _bootQinv = np.linalg.inv(_bootQ)
    if inactive is not None:
        _bootC = X_inactive.T.dot(_bootW.dot(X_active))
        _bootI = _bootC.dot(_bootQinv)

    noverall = active.sum()

    if inactive is not None:
        X_full = np.hstack([X_active,X_inactive])
    else:
        X_full = X_active

    def _boot_score(indices):
        X_star = X_full[indices]
        Y_star = Y[indices]
        score = X_star.T.dot(Y_star - _boot_mu(X_star))
        result = np.zeros(ntotal)
        result[:nactive] = _bootQinv.dot(score[:nactive])
        if ntotal > nactive:
            result[nactive:] = score[nactive:] + _bootI.dot(result[:nactive])
        return result

    return _boot_score, beta_active

def bootstrap_cov(m_n, boot_target, cross_terms=(), nsample=2000):
    """
    m out of n bootstrap
    """
    m, n = m_n

    _mean_target = 0.
    if len(cross_terms) > 0:
        _mean_cross = [0.] * len(cross_terms)
        _outer_cross = [0.] * len(cross_terms)
    _outer_target = 0.

    for _ in range(nsample):
        indices = np.random.choice(n, size=(m,), replace=True)
        _boot_target = boot_target(indices)

        _mean_target += _boot_target
        _outer_target += np.multiply.outer(_boot_target, _boot_target)

        for i, _boot in enumerate(cross_terms):
            _boot_sample = _boot(indices)
            _mean_cross[i] += _boot_sample
            _outer_cross[i] += np.multiply.outer(_boot_target, _boot_sample)

    _mean_target /= nsample
    _outer_target /= nsample

    for i in range(len(cross_terms)):
        _mean_cross[i] /= nsample
        _outer_cross[i] /= nsample

    _cov_target = _outer_target - np.multiply.outer(_mean_target, _mean_target)
    return [_cov_target] + [_o - np.multiply.outer(_mean_target, _m) for _m, _o in zip(_mean_cross, _outer_cross)]

def pairs_inactive_score_glm(glm_loss, active, beta_active):

    """
    Bootstrap inactive score at \bar{\beta}_E

    Will be used with forward stepwise.
    """
    inactive = ~active
    beta_full = np.zeros(glm_loss.shape)
    beta_full[active] = beta_active

    _full_boot_score = pairs_bootstrap_glm(glm_loss, 
                                           active, 
                                           beta_full=beta_full,
                                           inactive=inactive)[0]
    nactive = active.sum()
    def _boot_score(indices):
        return _full_boot_score(indices)[nactive:]

### test

from selection.algorithms.randomized import logistic_instance
from selection.sampling.randomized.randomization import base
from selection.sampling.langevin import projected_langevin
from selection.distributions.discrete_family import discrete_family

def main():
    s, n, p = 5, 200, 20 

    randomization = base.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = regreg_glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), np.ones(p)*lam)), lagrange=1.)

    # first randomization

    M_est1 = M_estimator(loss, epsilon, penalty, randomization)
    M_est1.solve()
    M_est1.setup_sampler()
    bootstrap_score1 = pairs_bootstrap_glm(M_est1.loss, 
                                           M_est1.overall, 
                                           beta_full=M_est1._beta_full, # this is private -- we "shouldn't" observe this
                                           inactive=M_est1.inactive)[0]

    # second randomization

    M_est2 = M_estimator(loss, epsilon, penalty, randomization)
    M_est2.solve()
    M_est2.setup_sampler()
    bootstrap_score2 = pairs_bootstrap_glm(M_est2.loss, 
                                           M_est2.overall, 
                                           beta_full=M_est2._beta_full, # this is private -- we "shouldn't" observe this
                                           inactive=M_est2.inactive)[0]

    # we take target to be union of two active sets

    active = M_est1.active + M_est2.active

    if set(nonzero).issubset(np.nonzero(active)[0]):
        boot_target, target_observed = pairs_bootstrap_glm(loss, active)

        # target are all true null coefficients selected

        target_cov, cov1, cov2 = bootstrap_cov((n, n), boot_target, cross_terms=(bootstrap_score1, bootstrap_score2))

        active_set = np.nonzero(active)[0]
        I = inactive_selected = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        
        A1, b1 = M_est1.condition(cov1[I], target_cov[I][:,I], target_observed[I])
        A2, b2 = M_est2.condition(cov2[I], target_cov[I][:,I], target_observed[I])

        target_inv_cov = np.linalg.inv(target_cov[I][:,I])

        initial_state = np.hstack([target_observed[I],
                                   M_est1._initial_opt_state,
                                   M_est2._initial_opt_state])


        ntarget = len(I)
        target_slice = slice(0, ntarget)
        opt_slice1 = slice(ntarget, p + ntarget)
        opt_slice2 = slice(p + ntarget, 2*p + ntarget)

        def target_gradient(state):
            # with many samplers, we will add up the `target_slice` component
            # many target_grads
            # and only once do the Gaussian addition of full_grad

            target = state[target_slice]
            opt_state1 = state[opt_slice1]
            opt_state2 = state[opt_slice2]
            target_grad1 = M_est1.gradient(target, (A1, b1), opt_state1)
            target_grad2 = M_est2.gradient(target, (A2, b2), opt_state2)

            full_grad = np.zeros_like(state)
            full_grad[opt_slice1] = -target_grad1[1]
            full_grad[opt_slice2] = -target_grad2[1]
            full_grad[target_slice] = -target_grad1[0] - target_grad2[0]

            full_grad[target_slice] -= target_inv_cov.dot(target)

            return full_grad

        def target_projection(state):
            opt_state1 = state[opt_slice1]
            state[opt_slice1] = M_est1.projection(opt_state1)
            opt_state2 = state[opt_slice2]
            state[opt_slice2] = M_est2.projection(opt_state2)
            return state

        target_langevin = projected_langevin(initial_state,
                                             target_gradient,
                                             target_projection,
                                             1. / p)


        Langevin_steps = 10000
        burning = 1000
        samples = []
        for i in range(Langevin_steps + burning):
            if (i>=burning):
                target_langevin.next()
                samples.append(target_langevin.state[target_slice].copy())
                
        test_stat = np.linalg.norm
        observed = test_stat(target_observed[I])
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.ccdf(0, observed)
        return pval
