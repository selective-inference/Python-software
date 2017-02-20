import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm as ndist
from scipy.stats import norm
import regreg.api as rr
from selection.randomized.api import randomization
from selection.bayesian.credible_intervals import projected_langevin

class nonnegative_softmax_scaled(rr.smooth_atom):
    """
    The nonnegative softmax objective
    .. math::
         \mu \mapsto
         \sum_{i=1}^{m} \log \left(1 +
         \frac{1}{\mu_i} \right)
    """

    objective_template = r"""\text{nonneg_softmax}\left(%(var)s\right)"""

    def __init__(self,
                 shape,
                 barrier_scale=1.,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        # a feasible point
        self.coefs[:] = np.ones(shape)
        self.barrier_scale = barrier_scale

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        slack = self.apply_offset(mean_param)

        if mode in ['both', 'func']:
            if np.all(slack > 0):
                f = self.scale(np.log((slack + self.barrier_scale) / slack).sum())
            else:
                f = np.inf
        if mode in ['both', 'grad']:
            g = self.scale(1. / (slack + self.barrier_scale) - 1. / slack)

        if mode == 'both':
            return f, g
        elif mode == 'grad':
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")


class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg)/(self.randomization_scale**2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg)/(self.randomization_scale**2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg>0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)
        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                        cube_prob[indicator]))/self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index])/
                                     ((prod_arg[pos_index]/arg_u[pos_index])-
                                      (1./arg_l[pos_index])))/self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] -(arg_l[neg_index]*neg_prod_arg[neg_index]))
                                    /self.randomization_scale)/(1.- neg_prod_arg[neg_index])


        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class selection_probability_objective(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,  # in R^n
                 noise_variance,
                 randomizer,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        """
        Objective function for $\beta_E$ (i.e. active) with $E$ the `active_set` optimization
        variables, and data $z \in \mathbb{R}^n$ (i.e. response).
        NEEDS UPDATING
        Above, $\beta_E^*$ is the `parameter`, $b_{\geq}$ is the softmax of the non-negative constraint,
        $$
        B_E = X^TX_E
        $$
        and
        $$
        \gamma_E = \begin{pmatrix} \lambda s_E\\ 0\end{pmatrix}
        $$
        with $\lambda$ being `lagrange`.
        Parameters
        ----------
        X : np.float
             Design matrix of shape (n,p)
        active : np.bool
             Boolean indicator of active set of shape (p,).
        active_signs : np.float
             Signs of active coefficients, of shape (active.sum(),).
        lagrange : np.float
             Array of lagrange penalties for LASSO of shape (p,)
        parameter : np.float
             Parameter $\beta_E^*$ for which we want to
             approximate the selection probability.
             Has shape (active_set.sum(),)
        randomization : np.float
             Variance of IID Gaussian noise
             that was added before selection.
        """

        n, p = X.shape
        E = active.sum()
        self.q = p - E
        self._X = X
        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        initial = np.zeros(n + E, )
        initial[n:] = feasible_point

        rr.smooth_atom.__init__(self,
                                (n + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.active = active
        nonnegative = nonnegative_softmax_scaled(E)  # should there be a
        # scale to our softmax?
        opt_vars = np.zeros(n + E, bool)
        opt_vars[n:] = 1

        self._opt_selector = rr.selector(opt_vars, (n + E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + E,))

        X_E = self.X_E = X[:, active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([-X[:, active].T, (B_E + epsilon * np.identity(E)) * active_signs[None, :]])
        self.A_inactive = np.hstack([-X[:, ~active].T, (B_mE * active_signs[None, :])])

        self.offset_active = active_signs * lagrange[active]

        # defines \gamma and likelihood loss
        self.set_parameter(mean_parameter, noise_variance)

        self.inactive_subgrad = np.zeros(p - E)

        self.active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                                 rr.affine_transform(self.A_active, self.offset_active))

        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = 1.)

        self.cube_loss = rr.affine_smooth(cube_obj, self.A_inactive)

        self.total_loss = rr.smooth_sum([self.active_conj_loss,
                                         self.cube_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier])

    def set_parameter(self, mean_parameter, noise_variance):
        """
        Set $\beta_E^*$.
        """
        mean_parameter = np.squeeze(mean_parameter)
        likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)
        self.likelihood_loss = rr.affine_smooth(likelihood_loss, self._response_selector)

    def smooth_objective(self, param, mode='both', check_feasibility=False):
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        param = self.apply_offset(param)

        if mode == 'func':
            f = self.total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = self.total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        n, p = self._X.shape

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current) * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.all(proposal[n:] > 0):
                    break
                step *= 0.5
                if count >= 40:
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                # print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        # print('iter', itercount)
        value = objective(current)
        return current, value


class sel_prob_gradient_map(rr.smooth_atom):
    def __init__(self,
                 X,
                 primal_feasible,
                 dual_feasible,
                 active,
                 active_signs,
                 lagrange,
                 generative_X,
                 noise_variance,
                 randomizer,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.E = active.sum()
        self.n, self.p = X.shape
        self.dim = generative_X.shape[1]

        self.noise_variance = noise_variance

        (self.X, self.primal_feasible, self.dual_feasible, self.active, self.active_signs, self.lagrange,
         self.generative_X, self.noise_variance, self.randomizer, self.epsilon) = \
            (X, primal_feasible, dual_feasible, active, active_signs, lagrange, generative_X, noise_variance,
             randomizer, epsilon)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):
        true_param = self.apply_offset(true_param)

        mean_parameter = np.squeeze(self.generative_X.dot(true_param))

        primal_sol = selection_probability_objective(self.X,
                                                     self.primal_feasible,
                                                     self.active,
                                                     self.active_signs,
                                                     self.lagrange,
                                                     mean_parameter,
                                                     self.noise_variance,
                                                     self.randomizer,
                                                     self.epsilon)

        sel_prob_primal = primal_sol.minimize2(nstep=60)[::-1]
        optimal_primal = (sel_prob_primal[1])[:self.n]
        sel_prob_val = -sel_prob_primal[0]
        optimizer = self.generative_X.T.dot(np.true_divide(optimal_primal - mean_parameter, self.noise_variance))

        if mode == 'func':
            return sel_prob_val
        elif mode == 'grad':
            return optimizer
        elif mode == 'both':
            return sel_prob_val, optimizer
        else:
            raise ValueError('mode incorrectly specified')


class selective_map_credible(rr.smooth_atom):
    def __init__(self,
                 y,
                 X,
                 primal_feasible,
                 dual_feasible,
                 active,
                 active_signs,
                 lagrange,
                 generative_X,
                 noise_variance,
                 prior_variance,
                 randomizer,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        self.param_shape = generative_X.shape[1]

        y = np.squeeze(y)

        self.E = active.sum()

        self.generative_X = generative_X

        if self.param_shape == self.E:
            initial = np.squeeze(primal_feasible * active_signs[None,:])
        else:
            initial = np.zeros(self.param_shape)

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.initial_state = initial

        self.set_likelihood(y, noise_variance, generative_X)

        self.set_prior(prior_variance)

        self.log_sel_prob = sel_prob_gradient_map(X,
                                                  primal_feasible,
                                                  dual_feasible,
                                                  active,
                                                  active_signs,
                                                  lagrange,
                                                  generative_X,
                                                  noise_variance,
                                                  randomizer,
                                                  epsilon)

        self.total_loss = rr.smooth_sum([self.likelihood_loss,
                                         self.log_prior_loss,
                                         self.log_sel_prob])

        self.noise_variance = noise_variance

    def set_likelihood(self, y, noise_variance, generative_X):
        likelihood_loss = rr.signal_approximator(y, coef=1. / noise_variance)
        self.likelihood_loss = rr.affine_smooth(likelihood_loss, generative_X)

    def set_prior(self, prior_variance):
        self.log_prior_loss = rr.signal_approximator(np.zeros(self.param_shape), coef=1. / prior_variance)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False):

        true_param = self.apply_offset(true_param)

        if mode == 'func':
            f = self.total_loss.smooth_objective(true_param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss.smooth_objective(true_param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = self.total_loss.smooth_objective(true_param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def map_solve_2(self, step=1, nstep=500, tol=1.e-8):

        current = self.coefs[:]
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current)* self.noise_variance

            # make sure proposal is a descent
            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)

                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        value = objective(current)
        return current, value

    def map_solve(self, initial=None, min_its=10, max_its=50, tol=1.e-10):

        problem = rr.separable_problem.singleton(self)
        #problem.coefs[:] = 0.5
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        value = problem.objective(soln)
        return soln, value

    def posterior_samples(self, Langevin_steps = 3000, burnin = 100):
        state = self.initial_state
        gradient_map = lambda x: -self.smooth_objective(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.E
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in range(Langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            #print i, sampler.state.copy()
            print i

        samples = np.array(samples)
        return samples[burnin:, :]