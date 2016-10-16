import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm as ndist

from selection.algorithms.softmax import nonnegative_softmax
import regreg.api as rr

#################################################################

### For arbitrary randomizations,
### we need at least the gradient of the
### CGF 

def cube_subproblem(argument, 
                    randomization_CGF_conjugate,
                    lagrange, nstep=100,
                    initial=None,
                    lipschitz=0):
    '''
    Solve the subproblem
    $$
    \text{minimize}_{z} \Lambda_{-E}^*(u + z_{-E}) + b_{-E}(z)
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
    if initial is None:
        current = np.zeros(k, np.float)
    else:
        current = initial # no copy

    current_value = np.inf

    conj_value, conj_grad = randomization_CGF_conjugate

    step = np.ones(k, np.float)

    objective = lambda u: cube_barrier(u, lagrange) + conj_value(argument + u)
        
    for itercount in range(nstep):
        newton_step = ((cube_gradient(current, lagrange) +
                        conj_grad(argument + current)) / 
                       (cube_hessian(current, lagrange) + lipschitz))

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * newton_step
            failing = (proposal > lagrange) + (proposal < - lagrange)
            if not failing.sum():
                break
            step *= 0.5**failing

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

        if np.fabs(current_value - proposed_value) < 1.e-6 * np.fabs(current_value):
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    value = objective(current)
    return current, value

def cube_barrier(argument, lagrange):
    '''
    Barrier approximation to the
    cube $[-\lambda,\lambda]^k$ with $\lambda$ being `lagrange`.
    The function is
    $$
    z \mapsto \log(1 + 1 / (\lambda - z)) + \log(1 + 1 / (z + \lambda))
    $$
    with $z$ being `argument`
    '''
    BIG = 10**10 # our Newton method will never evaluate this
                 # with any violations, but `scipy.minimize` does
    _diff = argument - lagrange # z - \lambda < 0
    _sum = argument + lagrange  # z + \lambda > 0
    violations = ((_diff >= 0).sum() + (_sum <= 0).sum() > 0)
    return np.log((_diff - 1.) * (_sum + 1.) / (_diff * _sum)).sum() + BIG * violations

def cube_gradient(argument, lagrange):
    """
    Gradient of approximation to the
    cube $[-\lambda,\lambda]^k$ with $\lambda$ being `lagrange`.

    The function is
    $$
    z \mapsto \frac{2}{\lambda - z} - \frac{1}{\lambda - z + 1} + 
    \frac{1}{z - \lambda + 1} 
    $$
    with $z$ being `argument`
    """
    _diff = argument - lagrange # z - \lambda < 0
    _sum = argument + lagrange  # z + \lambda > 0
    return 1. / (_diff - 1) - 1. / _diff + 1. / (_sum + 1) - 1. / _sum

def cube_hessian(argument, lagrange):
    """
    (Diagonal) Heissian of approximation to the
    cube $[-\lambda,\lambda]^k$ with $\lambda$ being `lagrange`.

    The function is
    $$
    z \mapsto \frac{2}{\lambda - z} - \frac{1}{\lambda - z + 1} + 
    \frac{1}{z - \lambda + 1} 
    $$
    with $z$ being `argument`
    """
    _diff = argument - lagrange # z - \lambda < 0
    _sum = argument + lagrange  # z + \lambda > 0
    return 1. / _diff**2 - 1. / (_diff - 1)**2 + 1. / _sum**2 - 1. / (_sum + 1)**2

class selection_probability_objective(rr.smooth_atom):

    def __init__(self, 
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter, # in R^n
                 noise_variance,
                 randomizer,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None):

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

        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError('randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        initial = np.zeros(n + E,)
        initial[n:] = feasible_point

        rr.smooth_atom.__init__(self,
                                (n + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.active = active
        nonnegative = nonnegative_softmax(E) # should there be a 
                                             # scale to our softmax?
        opt_vars = np.zeros(n+E, bool)
        opt_vars[n:] = 1

        self._opt_selector = rr.selector(opt_vars, (n+E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (n+E,))

        X_E = self.X_E = X[:,active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([-X[:,active].T, (B_E + epsilon * np.identity(E)) * active_signs[None,:]])
        self.A_inactive = np.hstack([-X[:,~active].T, (B_mE * active_signs[None,:])])

        self.offset_active = active_signs * lagrange[active]

        # defines \gamma and likelihood loss
        self.set_parameter(mean_parameter, noise_variance)

        self.inactive_subgrad = np.zeros(p - E)

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

        # as argument to convex conjugate of
        # inactive cube barrier
        # _i for inactive

        conjugate_value_i, barrier_gradient_i = self.cube_objective(param)
        f_active_conj, g_active_conj = self.active_conjugate_objective(param)

        if mode == 'func':
            f_nonneg = self.nonnegative_barrier.smooth_objective(param, 'func')
            f_like = self.likelihood_loss.smooth_objective(param, 'func')
            f = self.scale(f_nonneg + f_like + f_active_conj + conjugate_value_i)
            return f
        elif mode == 'grad':
            g_nonneg = self.nonnegative_barrier.smooth_objective(param, 'grad')
            g_like = self.likelihood_loss.smooth_objective(param, 'grad')
            g = self.scale(g_nonneg + g_like + g_active_conj + barrier_gradient_i)
            return g
        elif mode == 'both':
            f_nonneg, g_nonneg = self.nonnegative_barrier.smooth_objective(param, 'both')
            f_like, g_like = self.likelihood_loss.smooth_objective(param, 'both')
            f = self.scale(f_nonneg + f_like + f_active_conj + conjugate_value_i)
            g = self.scale(g_nonneg + g_like + g_active_conj + barrier_gradient_i)
            return f, g
        else:
            raise ValueError("mode incorrectly specified")

    def active_conjugate_objective(self, param):

        active_conj_value, active_conj_grad = self.active_conjugate
        param_a = self.A_active.dot(param)
        f_active_conj = active_conj_value(param_a + self.offset_active)
        g_active_conj = self.A_active.T.dot(active_conj_grad(param_a+self.offset_active))

        return f_active_conj, g_active_conj

    def cube_objective(self, param):

        conjugate_argument_i = self.A_inactive.dot(param)
        conjugate_optimizer_i, conjugate_value_i = cube_subproblem(conjugate_argument_i,
                                                                   self.inactive_conjugate,
                                                                   self.inactive_lagrange,
                                                                   initial=self.inactive_subgrad)

        barrier_gradient_i = self.A_inactive.T.dot(conjugate_optimizer_i)
        return conjugate_value_i, barrier_gradient_i

    def minimize(self, initial=None):

        nonneg_con = self._opt_selector.output_shape[0]
        constraint = rr.separable(self.shape,
                                  [rr.nonnegative((nonneg_con,), offset=1.e-12 * np.ones(nonneg_con))],
                                  [self._opt_selector.index_obj])

        problem = rr.separable_problem.fromatom(constraint, self)
        problem.coefs[self._opt_selector.index_obj] = 0.5
        soln = problem.solve(max_its=200, min_its=20, tol=1.e-12)
        value = problem.objective(soln)
        return soln, value





































