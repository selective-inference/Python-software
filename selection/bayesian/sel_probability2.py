import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm as ndist

from selection.algorithms.softmax import nonnegative_softmax
import regreg.api as rr

#########################################################
#####defining a class for computing selection probability: also returns selective_map and gradient of posterior
class selection_probability(object):

    # defining class variables
    def __init__(self, V, B_E, gamma_E, sigma, tau, lam, y, betaE, cube):

        (self.V, self.B_E, self.gamma_E, self.sigma, self.tau, self.lam, self.y, self.betaE, self.cube) = (V, B_E,
                                                                                                           gamma_E,
                                                                                                           sigma, tau,
                                                                                                           lam, y,betaE,
                                                                                                           cube)
        self.sigma_sq = self.sigma ** 2
        self.tau_sq = self.tau ** 2
        self.signs = np.sign(self.betaE)
        self.n = self.y.shape[0]
        self.p = self.B_E.shape[0]
        self.nactive = self.betaE.shape[0]
        self.ninactive = self.p - self.nactive
        # for lasso, V=-X, B_E=\begin{pmatrix} X_E^T X_E+\epsilon I & 0 \\ X_{-E}^T X_E & I \end{pmatrix}, gamma_E=
        # \begin{pmatrix} \lambda* s_E \\ 0\end{pamtrix}

        # be careful here to permute the active columns beforehand as code
        # assumes the active columns in the first |E| positions
        self.V_E = self.V[:, :self.nactive]
        self.V_E_comp = self.V[:, self.nactive:]
        self.C_E = self.B_E[:self.nactive, :self.nactive]
        self.D_E = self.B_E.T[:self.nactive, self.nactive:]
        self.Sigma = np.true_divide(np.identity(self.n), self.sigma_sq) + np.true_divide(
            np.dot(self.V, self.V.T), self.tau_sq)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.Sigma_inter = np.true_divide(np.identity(self.p), self.tau_sq) - np.true_divide(np.dot(np.dot(
            self.V.T, self.Sigma_inv), self.V), self.tau_sq ** 2)
        self.constant=np.true_divide(np.dot(np.dot(self.V_E.T, self.Sigma_inv), self.V_E), self.sigma_sq**2)
        self.mat_inter = -np.dot(np.true_divide(np.dot(self.B_E.T, self.V.T), self.tau_sq), self.Sigma_inv)
        self.Sigma_noise = np.dot(np.dot(self.B_E.T, self.Sigma_inter), self.B_E)
        self.vec_inter = np.true_divide(np.dot(self.B_E.T, self.gamma_E), self.tau_sq)
        self.mu_noise = np.dot(self.mat_inter, - np.true_divide(np.dot(self.V, self.gamma_E),
                                                                self.tau_sq)) - self.vec_inter
        self.mu_coef = np.true_divide(-self.lam * np.dot(self.C_E, self.signs), self.tau_sq)
        self.Sigma_coef = np.true_divide(np.dot(self.C_E, self.C_E) + np.dot(self.D_E, self.D_E.T), self.tau_sq)
        self.mu_data = - np.true_divide(np.dot(self.V, self.gamma_E),self.tau_sq)

    # defining log prior to be the Gaussian prior
    def log_prior(self, param, gamma):
        return -np.true_divide(np.linalg.norm(param) ** 2, 2*(gamma ** 2))

    def optimization(self, param):

        if self.p < self.n + self.nactive:
            initial_noise = np.zeros(self.p)
            initial_noise[:self.nactive] = self.betaE
            initial_noise[self.nactive:] = self.cube
            res = minimize(objective_noise, x0=initial_noise)
            const_param = np.dot(np.dot(param.T,self.constant),param)
            return -res.fun+const_param, res.x
        else:
            initial_data = self.y
            res = minimize(objective_data, x0=initial_data)
            return -res.fun, res.x


    def selective_map(self,y,prior_sd):
        def objective(param,y,prior_sd):
            return -np.true_divide(np.dot(y.T,-np.dot(self.V_E, param)),
                              self.sigma_sq)-self.log_prior(param,prior_sd)+self.optimization(param)[0]
        map_prob=minimize(objective,x0=self.betaE,args=(y,prior_sd))
        return map_prob.x

    def gradient(self,param,y,prior_sd):
        if self.p< self.n+self.nactive:
            func_param=np.dot(self.constant,param)
            grad_sel_prob= np.dot(np.dot(self.mat_inter, -np.true_divide(self.V_E, self.sigma_sq)).T,
                                  self.optimization(param)[1])+func_param
        else:
            grad_sel_prob= np.dot(-np.true_divide(self.V_E.T, self.sigma_sq),self.optimization(param)[1])

        return np.true_divide(-np.dot(self.V_E.T,y),self.sigma_sq) -np.true_divide(param,prior_sd**2)-grad_sel_prob

# defining barrier function on betaE
def barrier_sel(z_2):
    # A_betaE beta_E\leq 0
    A_betaE = -np.diag(self.signs)
    if all(- np.dot(A_betaE, z_2) >= np.power(10, -9)):
        return np.sum(np.log(1 + np.true_divide(1, - np.dot(A_betaE, z_2))))
    return self.nactive * np.log(1 + 10 ** 9)

# defining barrier function on u_{-E}
def barrier_subgrad(z_3):

    # A_2 beta_E\leq b
    A_subgrad = np.zeros(((2 * self.ninactive), (self.ninactive)))
    A_subgrad[:self.ninactive, :] = np.identity(self.ninactive)
    A_subgrad[self.ninactive:, :] = -np.identity(self.ninactive)
    b = np.ones((2 * self.ninactive))
    if all(b - np.dot(A_subgrad, z_3) >= np.power(10, -9)):
        return np.sum(np.log(1 + np.true_divide(1, b - np.dot(A_subgrad, z_3))))
    return b.shape[0] * np.log(1 + 10 ** 9)

def barrier_subgrad_coord(z):
    # A_2 beta_E\leq b
    # a = np.array([1,-1])
    # b = np.ones(2)
    if -1 + np.power(10, -9) < z < 1 - np.power(10, -9):
        return np.log(1 + np.true_divide(1, 1 - z)) + np.log(1 + np.true_divide(1, 1 + z))
    return 2 * np.log(1 + 10 ** 9)

#defining objective function in p dimensions to be optimized when p<n+|E|
def objective_noise(z):

    z_2 = z[:self.nactive]
    z_3 = z[self.nactive:]
    mu_noise_mod = self.mu_noise.copy()
    mu_noise_mod+=np.dot(self.mat_inter,np.true_divide(-np.dot(self.V_E, param), self.sigma_sq))
    return np.true_divide(np.dot(np.dot(z.T, self.Sigma_noise), z), 2)+barrier_sel(
        z_2)+barrier_subgrad(z_3)-np.dot(z.T, mu_noise_mod)

#defining objective in 3 steps when p>n+|E|, first optimize over u_{-E}
# defining the objective for subgradient coordinate wise
def obj_subgrad(z, mu_coord):
    return -(z * mu_coord) + np.true_divide(z ** 2, 2 * self.tau_sq) + barrier_subgrad_coord(z)

def value_subgrad_coordinate(z_1, z_2):
    mu_subgrad = np.true_divide(-np.dot(self.V_E_comp.T, z_1) - np.dot(self.D_E.T, z_2), self.tau_sq)
    res_seq=[]
    for i in range(self.ninactive):
        mu_coord=mu_subgrad[i]
        res=minimize(obj_subgrad, x0=self.cube[i], args=mu_coord)
        res_seq.append(-res.fun)
    return(np.sum(res_seq))

#defining objective over z_2
def objective_coef(z_2,z_1):
    mu_coef_mod=self.mu_coef.copy()- np.true_divide(np.dot(np.dot(
        self.C_E, self.V_E.T) + np.dot(self.D_E, self.V_E_comp.T), z_1),self.tau_sq)
    return - np.dot(z_2.T,mu_coef_mod) + np.true_divide(np.dot(np.dot(
        z_2.T,self.Sigma_coef),z_2),2)+barrier_sel(z_2)-value_subgrad_coordinate(z_1, z_2)

#defining objectiv over z_1
def objective_data(z_1):
    mu_data_mod = self.mu_data.copy()+ np.true_divide(-np.dot(self.V_E, param), self.sigma_sq)
    value_coef = minimize(objective_coef, x0=self.betaE, args=z_1)
    return -np.dot(z_1.T, mu_data_mod) + np.true_divide(np.dot(np.dot(z_1.T, self.Sigma), z_1),
                                                        2) + value_coef.fun





#################################################################


### For arbitrary randomizations,
### we need at least the gradient of the
### CGF 

def cube_subproblem(argument, 
                    randomization_CGF_conjugate,
                    lagrange, nstep=30,
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
                 randomization,
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
        self.randomization = randomization

        self.inactive_conjugate = self.active_conjugate = randomization.CGF_conjugate
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

        opt_selector = rr.selector(opt_vars, (n+E,))
        self.nonnegative_barrier = nonnegative.linear(opt_selector)
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
        likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)
        #likelihood_loss.quadratic = rr.identity_quadratic(0, 0, 0,
                                                         # -0.5 * (mean_parameter**2).sum() / noise_variance)
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

        conjugate_argument_i = self.A_inactive.dot(param)

        conjugate_optimizer_i, conjugate_value_i = cube_subproblem(conjugate_argument_i,
                                                                   self.inactive_conjugate,
                                                                   self.inactive_lagrange,
                                                                   initial=self.inactive_subgrad)

        barrier_gradient_i = self.A_inactive.T.dot(conjugate_optimizer_i)

        active_conj_value, active_conj_grad = self.active_conjugate

        if mode == 'func':
            f_nonneg = self.nonnegative_barrier.smooth_objective(param, 'func')
            f_like = self.likelihood_loss.smooth_objective(param, 'func')
            f_active_conj = active_conj_value(self.A_active.dot(param)+self.offset_active)
            f = self.scale(f_nonneg + f_like + f_active_conj + conjugate_value_i)
            #print(f, f_nonneg, f_like, f_active_conj, conjugate_value_i, 'value')
            return f
        elif mode == 'grad':
            g_nonneg = self.nonnegative_barrier.smooth_objective(param, 'grad')
            g_like = self.likelihood_loss.smooth_objective(param, 'grad')
            g_active_conj = self.A_active.T.dot(active_conj_grad(self.A_active.dot(param)+self.offset_active))
            g = self.scale(g_nonneg + g_like + g_active_conj + barrier_gradient_i)
            #print(g, 'grad')
            return g
        elif mode == 'both':
            f_nonneg, g_nonneg = self.nonnegative_barrier.smooth_objective(param, 'both')
            f_like, g_like = self.likelihood_loss.smooth_objective(param, 'both')
            param_a = self.A_active.dot(param)
            f_active_conj = active_conj_value(param_a+self.offset_active)
            g_active_conj = self.A_active.T.dot(active_conj_grad(param_a)+self.offset_active)
            f = self.scale(f_nonneg + f_like + f_active_conj + conjugate_value_i)
            g = self.scale(g_nonneg + g_like + g_active_conj + barrier_gradient_i)
            #print(f, f_nonneg, f_like, f_active_conj, conjugate_value_i, 'value')
            return f, g
        else:
            raise ValueError("mode incorrectly specified")

    def minimize(self, initial=None, step=1, nstep=30):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')

        for itercount in range(nstep):
            newton_step = self.smooth_objective(current, 'grad') * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.isfinite(objective(proposal)): 
                    break
                step *= 0.5
                if count >= 40:
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print(current_value, proposed_value, 'minimize')
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












































