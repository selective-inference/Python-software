import numpy as np

from selection.algorithms.softmax import nonnegative_softmax
import regreg.api as rr
from selection.bayesian.selection_probability_rr import cube_barrier_scaled, cube_gradient_scaled, cube_hessian_scaled
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.barrier_fs import linear_map, fs_conjugate, barrier_conjugate_fs_rr

def cube_subproblem_fs_linear(argument,
                              c,
                              randomization_CGF_conjugate,
                              nstep=100,
                              initial=None,
                              lipschitz=0,
                              tol=1.e-10):
    '''
    Solve the subproblem
    $$
    \text{minimize}_{z} \Lambda_{-E}^*(u + z_{-E}) + b_{-E}(z; c)
    $$
    where $u$ is `argument`, $\Lambda_{-E}^*$ is the
    conjvex conjugate of the $-E$ coordinates of the
    randomization (assumes that randomization has independent
    coordinates) and
    $b_{-E}$ is a barrier approximation to
    the cube $\prod_{j \in -E} [-\lambda_j,\lambda_j]$ with
    $\lambda$ being `lagrange`.
    Returns the maximizer and the value of the convex conjugate.
    '''

    k = argument.shape[0]
    lagrange = c * np.ones(k)
    if initial is None:
        current = np.zeros(k, np.float)
    else:
        current = initial  # no copy

    current_value = np.inf

    conj_value = lambda x: randomization_CGF_conjugate.smooth_objective(x, 'func')
    conj_grad = lambda x: randomization_CGF_conjugate.smooth_objective(x, 'grad')

    step = np.ones(k, np.float)
    objective = lambda u: cube_barrier_scaled(u, lagrange) + conj_value(argument + u)

    for itercount in range(nstep):
        newton_step = ((cube_gradient_scaled(current, lagrange) +
                        (conj_grad(argument + current))) /
                       (cube_hessian_scaled(current, lagrange) + lipschitz))

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * newton_step
            failing = (proposal > lagrange) + (proposal < - lagrange)
            if not failing.sum():
                break
            step *= 0.5 ** failing

            if count >= 40:
                raise ValueError('not finding a feasible point')

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

class cube_objective_fs_linear(rr.smooth_atom):
    def __init__(self,
                 randomization_CGF_conjugate,
                 nstep=100,
                 tol=1.e-10,
                 initial=None,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        (self.randomization_CGF_conjugate,
         self.nstep,
         self.tol) = (randomization_CGF_conjugate,
                      nstep,
                      tol)

        rr.smooth_atom.__init__(self,
                                randomization_CGF_conjugate.shape,
                                initial=initial,
                                coef=coef,
                                offset=offset,
                                quadratic=quadratic)

    def smooth_objective(self, arg, mode='both', check_feasibility=False):

        arg = self.apply_offset(arg)

        arg_shape = arg.shape[0]

        c_bool = np.zeros(arg_shape, bool)

        c_bool[(arg_shape-1):] = 1

        z = arg[~c_bool]

        c = arg[c_bool]

        optimizer, value = cube_subproblem_fs_linear(z,
                                                     c,
                                                     self.randomization_CGF_conjugate,
                                                     nstep=self.nstep,
                                                     tol=self.tol)

        #print "opt scaled",optimizer/6.

        gradient_z = z + optimizer

        lagrange = c* np.ones(z.shape[0])
        _diff_c = optimizer - lagrange # z - c < 0
        _sum_c = optimizer +  lagrange # z + c > 0

        _num = -2. / ((_diff_c - lagrange)**2) + 1. /( _diff_c**2) + 2. / ((_sum_c + lagrange)**2) \
                         - 1. / (_sum_c**2)

        _den = np.ones(z.shape[0]) - (-1. / _diff_c ** 2 + 1. / ((_diff_c - lagrange) ** 2) - 1. / _sum_c ** 2 + \
           1. / ((_sum_c + lagrange) ** 2))

        gradient_c_max = np.true_divide(_num,_den)

        vec_1 = 1./(_diff_c - lagrange) + 1./(_sum_c + lagrange) - 1./_diff_c -1./_sum_c

        vec_2 = (-2./(_diff_c- lagrange) + 2./(_sum_c + lagrange) + 1./_diff_c -1./_sum_c).sum()

        gradient_c = (z + optimizer + vec_1).T.dot(gradient_c_max) + vec_2

        #print gradient_z.shape, gradient_c.shape

        gradient = np.append(gradient_z, gradient_c)

        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(gradient)
        elif mode == 'both':
            return self.scale(value), self.scale(gradient)
        else:
            raise ValueError("mode incorrectly specified")


class selection_probability_objective_fs_rp(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_sign,
                 mean_parameter,  # in R^n
                 noise_variance,
                 randomizer,
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
        E = 1
        self._X = X
        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = np.zeros(n + E, )
        initial[n:] = feasible_point

        rr.smooth_atom.__init__(self,
                                (n + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        nonnegative = nonnegative_softmax(E)

        opt_vars = np.zeros(n + E, bool)
        opt_vars[n:] = 1

        self._opt_selector = rr.selector(opt_vars, (n + E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n + E,))

        sign_array = np.zeros((E,E))
        sign_array[0:,:] = active_sign

        #print sign_array.shape, X[:, active].T.shape, X[:, ~active].T.shape, np.zeros(p-E).shape
        self.A_active = np.hstack([-X[:, active].T, sign_array])
        self.A_inactive_1 = np.hstack([-X[:, ~active].T, np.zeros((p-E,1))])
        self.A_inactive_2 = np.hstack([np.zeros((n,E)).T, sign_array])
        self.A_inactive = np.vstack([self.A_inactive_1, self.A_inactive_2])

        #print self.A_active.shape, self.A_inactive.shape

        # defines \gamma and likelihood loss
        self.set_parameter(mean_parameter, noise_variance)

        self.active_conj_loss = rr.affine_smooth(self.active_conjugate, self.A_active)

        cube_obj = cube_objective_fs_linear(self.inactive_conjugate)

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

    def minimize(self, initial=None, min_its=10, max_its=50, tol=1.e-10):

        nonneg_con = self._opt_selector.output_shape[0]
        constraint = rr.separable(self.shape,
                                  [rr.nonnegative((nonneg_con,), offset=1.e-12 * np.ones(nonneg_con))],
                                  [self._opt_selector.index_obj])

        problem = rr.separable_problem.fromatom(constraint, self)
        problem.coefs[:] = 0.5
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        value = problem.objective(soln)
        return soln, value

    def minimize2(self, step=1, nstep=30, tol=1.e-6):

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

        #print('iter', itercount)
        value = objective(current)
        return current, value


class dual_selection_probability_fs(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_sign,
                 mean_parameter,  # in R^n
                 noise_variance,
                 randomizer,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        n, p = X.shape
        E = 1
        self._X = X
        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.CGF_randomization = randomizer.CGF

        if self.CGF_randomization is None:
            raise ValueError(
                'randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        initial = feasible_point

        self.feasible_point = feasible_point

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = feasible_point

        mean_parameter = np.squeeze(mean_parameter)

        self.active = active

        sign_array = np.zeros((E, E))
        sign_array[0:, :] = active_sign

        self.X_E = X[:, active]
        self.X_permute = np.hstack([self.X_E, self._X[:, ~active]])

        self.B_active = np.hstack([sign_array, np.zeros((E, p - E))])
        self.B_inactive = np.hstack([np.zeros((p-E,E)), np.identity((p - E))])
        self.B_p_inv = np.vstack((self.B_active, self.B_inactive))

        self.cube_bool = np.zeros(p, np.bool)

        self.cube_bool[E:] = 1

        self._opt_selector = rr.selector(~self.cube_bool, (p,))

        self.set_parameter(mean_parameter, noise_variance)

        _barrier_star = barrier_conjugate_fs_rr(self.cube_bool)

        # _barrier_star = barrier_conjugate_log(self.cube_bool, self.inactive_lagrange)

        self.conjugate_barrier = _barrier_star

        #self.conjugate_barrier = rr.affine_smooth(_barrier_star, np.identity(p))

        print self.conjugate_barrier.smooth_objective(np.append(np.append(-1, 0.5*np.ones(p-2)),0.9))

        self.CGF_randomizer = rr.affine_smooth(self.CGF_randomization, -self.B_p_inv)

        self.constant = np.true_divide(mean_parameter.dot(mean_parameter), 2 * noise_variance)

        self.linear_term = rr.identity_quadratic(0, 0, 0, -self.constant)

        self.total_loss = rr.smooth_sum([self.conjugate_barrier,
                                         self.CGF_randomizer,
                                         self.likelihood_loss])

        self.total_loss.quadratic = self.linear_term

    def set_parameter(self, mean_parameter, noise_variance):

        mean_parameter = np.squeeze(mean_parameter)

        self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)

        self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, self.X_permute.dot(self.B_p_inv))

    def _smooth_objective(self, param, mode='both', check_feasibility=False):

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

    def minimize(self, initial=None, min_its=10, max_its=50, tol=1.e-10):

        nonpos_con = self._opt_selector.output_shape[0]
        constraint = rr.separable(self.shape,
                                  [rr.nonpositive((nonpos_con,), offset=-1.e-12 * np.ones(nonpos_con))],
                                  [self._opt_selector.index_obj])

        problem = rr.separable_problem.fromatom(constraint, self.total_loss)
        problem.coefs[:] = self.coefs
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        self.coefs[:] = soln
        value = problem.objective(soln)
        return soln, value

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        n, p = self._X.shape

        current = self.feasible_point
        current_value = np.inf

        objective = lambda u: self.total_loss.objective(u)
        grad = lambda u: self.total_loss.smooth_objective(u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current) * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.all(proposal[:1] < 0):
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
