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

        X, Y = loss.data
        if loss._is_transform:
            raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
        X_restricted = X[:,overall]
        loss_restricted = rr.affine_smooth(loss.loss, X_restricted)
        _beta_unpenalized = loss_restricted.solve(**solve_args)
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

    def form_covariance(self, target):
        """
        For an estimator of a target statistical 
        functional, compute covariance
        of the estimator with score.
        """
        raise NotImplementedError('abstract method')

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


    def condition(self, target_score_cov, target_cov, initial_target_state):
        """
        condition the score on the target,
        return a new score_transform
        that is composition of `self.score_transform`
        with the affine map from conditioning
        """

        target_score_cov = np.atleast_2d(target_score_cov) 
        target_cov = np.atleast_2d(target_cov) 
        initial_target_state = np.atleast_1d(initial_target_state)

        linear_part = target_score_cov.T.dot(np.linalg.pinv(target_cov))
        offset = self._initial_score_state - linear_part.dot(initial_target_state)

        # now compute the composition of this map with
        # self.score_transform

        score_linear, score_offset = self.score_transform
        composition_linear_part = score_linear.dot(linear_part)
        composition_offset = score_linear.dot(offset) + score_offset

        return (composition_linear_part, composition_offset)
        


class glm(M_estimator):

    def form_covariance(self, bootstrap_target, nsample=2000):
        """

        """
        self.setup_bootstrap()

        _target_mean = 0.
        _score_mean = 0.
        _cov = 0.
        for _ in range(nsample):
            indices = np.random.choice(n, size=(n,), replace=True)
            target_star = bootstrap_target(indices)
            score_star = self.bootstrap_score(indices)

            _target_mean += target_star
            _score_mean += score_star

            _cov += np.multiply.outer(target_star, score_star)

        _cov /= nsample
        _target_mean = _target_mean / nsample
        _score_mean = _score_mean / nsample
        _cov -= np.multiply.outer(_target_mean, _score_mean)

        return _cov

    def setup_bootstrap(self):
        """
        Should define a callable _boot_score
        that takes `indices` and returns
        a bootstrap sample of (\bar{\beta}_{E \cup U}, N_{-(E \cup U)})
        """
        # form objects needed to bootstrap the score
        # which will be used to estimate covariance
        # of score and target model parameters

        # this code below will work for GLMs
        # not general M-estimators !!!
        # self.loss is the loss for a saturated GLM likelihood

        # gradient of saturated loss if \mu - Y

        overall, inactive = self.overall, self.inactive

        X, Y = self.loss.data

        if isinstance(self.loss.loss, logistic_loglike):
            Y = Y[0]
        self._bootX, self._bootY = X, Y

        _boot_mu = lambda X: self.loss.loss.smooth_objective(X.dot(self._beta_full), 'grad') + Y

        _bootQ = np.zeros((p,p))

        _bootW = np.diag(self.loss.loss.hessian(X.dot(self._beta_full)))
        _bootQ = X[:, overall].T.dot(_bootW.dot(X[:, overall]))
        _bootQinv = np.linalg.inv(_bootQ)
        _bootC = X[:, inactive].T.dot(_bootW.dot(X[:, overall]))
        _bootI = _bootC.dot(_bootQinv)

        noverall = overall.sum()
        def _boot_score(X_star, Y_star):
            score = X_star.T.dot(Y_star - _boot_mu(X_star))
            result = np.zeros_like(score)
            result[:noverall] = _bootQinv.dot(score[overall])
            result[noverall:] = score[inactive] + _bootI.dot(result[:noverall])
            return result
        self._boot_score = _boot_score

    def bootstrap_score(self, indices):
        """
        """

        if not hasattr(self, "_boot_score"):
            raise ValueError('setup_bootstrap should be called before using this function')

        X, Y = self._bootX, self._bootY
        Y_star = Y[indices]
        X_star = X[indices]

        return self._boot_score(X_star, Y_star)

    def bootstrap_target(self, indices):
        """
        Bootstrap the `overall` M-estimator coefficients
        """
        overall = self.overall
        return self.bootstrap_score(indices)[:overall.sum()]

    def form_target_cov(self, nsample=2000):
        _mean = 0.
        _crossprod = 0

        for _ in range(nsample):
            indices = np.random.choice(n, size=(n,), replace=True)
            _target = self.bootstrap_target(indices)
            _mean += _target
            _crossprod += np.multiply.outer(_target, _target)

        _mean /= nsample
        _crossprod /= nsample
        return _crossprod - np.multiply.outer(_mean, _mean)

if __name__ == "__main__":

    from selection.algorithms.randomized import logistic_instance
    from selection.sampling.randomized.randomization import base
    from selection.sampling.langevin import projected_langevin

    s, n, p = 5, 200, 20 

    randomization = base.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)
    print 'true_beta', beta
    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = regreg_glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), np.ones(p)*lam)), lagrange=1.)

    # first randomization

    M_est = glm(loss, epsilon, penalty, randomization)
    M_est.solve()
    M_est.setup_sampler()

    # second randomization

    M_est2 = glm(loss, epsilon, penalty, randomization)
    M_est2.solve()
    M_est2.setup_sampler()

    # for exposition, we will just take
    # the target from first randomization
    # should really do something different 

    cov = M_est.form_covariance(M_est.bootstrap_target)
    cov2 = M_est2.form_covariance(M_est.bootstrap_target)
    target_cov = M_est.form_target_cov()

    print cov.shape, target_cov.shape

    target_initial = M_est._initial_score_state[:M_est.active.sum()]

    # for second coefficient
    A1, b1 = M_est.condition(cov[1], target_cov[1,1], target_initial[1])
    A2, b2 = M_est2.condition(cov2[1], target_cov[1,1], target_initial[1])

    target_inv_cov = 1. / target_cov[1,1]

    initial_state = np.hstack([target_initial[1],
                               M_est._initial_opt_state,
                               M_est2._initial_opt_state])

    target_slice = slice(0,1)
    opt_slice = slice(1, p+1)
    opt_slice2 = slice(p+1, 2*p+1)

    def target_gradient(state):
        # with many samplers, we will add up the `target_slice` component
        # many target_grads
        # and only once do the Gaussian addition of full_grad

        target = state[target_slice]
        opt_state = state[opt_slice]
        opt_state2 = state[opt_slice2]
        target_grad = M_est.gradient(target, (A1, b1), opt_state)
        target_grad2 = M_est.gradient(target, (A2, b2), opt_state2)

        full_grad = np.zeros_like(state)
        full_grad[opt_slice] = target_grad[1]
        full_grad[opt_slice2] = target_grad2[1]
        full_grad[target_slice] = target_grad[0] + target_grad2[0]

        full_grad[target_slice] -= target / target_cov[1,1]

        return full_grad

    def target_projection(state):
        opt_state = state[opt_slice]
        state[opt_slice] = M_est.projection(opt_state)
        opt_state2 = state[opt_slice2]
        state[opt_slice2] = M_est2.projection(opt_state2)
        return state

    target_langevin = projected_langevin(initial_state,
                                         target_gradient,
                                         target_projection,
                                         1. / p)


    Langevin_steps = 1000
    burning = 100
    samples = []
    for i in range(Langevin_steps):
        if (i>burning):
            target_langevin.next()
            samples.append(target_langevin.state[target_slice].copy())
