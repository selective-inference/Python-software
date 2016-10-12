from selection.algorithms.softmax import nonnegative_softmax
import regreg.api as rr
import numpy as np
from scipy.optimize import minimize
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.sel_probability2 import cube_subproblem, cube_gradient, cube_barrier
from selection.bayesian.barrier import barrier_conjugate

class my_selection_probability_only_objective(object):

    # defining class variables
    def __init__(self, V, B_E, gamma_E, sigma, tau, lam, y, betaE, cube, vec, active_coef):

        (self.V, self.B_E, self.gamma_E, self.sigma, self.tau, self.lam, self.y, self.betaE, self.cube, self.vec,
         self.active_coef) = (V, B_E, gamma_E, sigma, tau, lam, y,betaE, cube, vec, active_coef)
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
                return np.log(1 + np.true_divide(1, (self.lam*(1 - z)))) + np.log(1 + np.true_divide(1,(self.lam*(1 + z))))
            return 2 * np.log(1 + np.true_divide(10 ** 9,self.lam))

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
            return -(self.lam*(z * mu_coord)) + ((self.lam**2)*np.true_divide(z ** 2, 2 * self.tau_sq))\
                   + barrier_subgrad_coord(z)

        def value_subgrad_coordinate(z_1, z_2):
            mu_subgrad = np.true_divide(-np.dot(self.V_E_comp.T, z_1) - np.dot(self.D_E.T, z_2), self.tau_sq)
            res_seq=[]
            for i in range(self.ninactive):
                mu_coord=mu_subgrad[i]
                res=minimize(obj_subgrad, x0=self.cube[i], args=mu_coord)
                res_seq.append(-res.fun)
            return np.sum(res_seq)

        #defining objective over z_2
        def objective_coef(z_2,z_1):
            mu_coef_mod=self.mu_coef.copy()- np.true_divide(np.dot(np.dot(
                self.C_E, self.V_E.T) + np.dot(self.D_E, self.V_E_comp.T), z_1),self.tau_sq)
            return - np.dot(z_2.T,mu_coef_mod) + np.true_divide(np.dot(np.dot(
                z_2.T,self.Sigma_coef),z_2),2)+ barrier_sel(z_2)

        #defining objective over z_1
        def objective_data(z_1):
            mu_data_mod = self.mu_data.copy()+ np.true_divide(-np.dot(self.V_E, param), self.sigma_sq)
            value_coef = objective_coef(self.active_coef,z_1)
            return -np.dot(z_1.T, mu_data_mod) + np.true_divide(np.dot(np.dot(z_1.T, self.Sigma), z_1), 2) + value_coef

        return objective_data(self.vec), value_subgrad_coordinate(self.vec, self.active_coef)


class selection_probability_only_objective(rr.smooth_atom):
    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,  # in R^n
                 noise_variance,
                 randomization,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        n, p = X.shape
        E = active.sum()

        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomization

        self.inactive_conjugate = self.active_conjugate = randomization.CGF_conjugate
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
        nonnegative = nonnegative_softmax(E)  # should there be a
        # scale to our softmax?
        opt_vars = np.zeros(n + E, bool)
        opt_vars[n:] = 1

        opt_selector = rr.selector(opt_vars, (n + E,))
        self.nonnegative_barrier = nonnegative.linear(opt_selector)
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

    def set_parameter(self, mean_parameter, noise_variance):
        """
        Set $\beta_E^*$.
        """
        likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)
        self.likelihood_loss = rr.affine_smooth(likelihood_loss, self._response_selector)

    def smooth_objective(self, param, mode='func', check_feasibility=False):

        param = self.apply_offset(param)

        conjugate_argument_i = self.A_inactive.dot(param)

        conjugate_optimizer_i, conjugate_value_i = cube_subproblem(conjugate_argument_i,
                                                                   self.inactive_conjugate,
                                                                   self.inactive_lagrange,
                                                                   initial=self.inactive_subgrad)

        constant = np.true_divide(np.dot(conjugate_argument_i.T, conjugate_argument_i), 2)

        active_conj_value, active_conj_grad = self.active_conjugate

        if mode == 'func':
            f_nonneg = self.nonnegative_barrier.smooth_objective(param, 'func')
            f_like = self.likelihood_loss.smooth_objective(param, 'func')
            f_active_conj = active_conj_value(self.A_active.dot(param) + self.offset_active)
            return f_nonneg + f_like + f_active_conj + constant, -conjugate_value_i + constant


class dual_selection_probability_only_objective(rr.smooth_atom):

    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,
                 noise_variance,
                 randomization,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.X=X
        self.feasible_point = feasible_point
        n, p = X.shape
        E = active.sum()

        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomization

        self.CGF_randomization = randomization.CGF

        if self.CGF_randomization is None:
            raise ValueError('randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        X_E = self.X_E = X[:,active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :],np.zeros((E,p-E))])
        self.A_inactive = np.hstack([B_mE * active_signs[None, :],np.identity((p-E))])
        self.A=np.vstack((self.A_active,self.A_inactive))
        self.A_E = self.A.T[:E,:]
        self.dual_arg = np.zeros(p)
        self.dual_arg[:E] = -active_signs * lagrange[active]
        self.feasible_point=feasible_point


        initial=feasible_point

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.cube_bool = np.zeros(p, np.bool)
        self.cube_bool[E:] = 1

        self.set_parameter(mean_parameter, noise_variance)

        self.coefs[:] = initial

    def set_parameter(self, mean_parameter, noise_variance):

        self.likelihood_loss = rr.signal_approximator(mean_parameter, coef=1. / noise_variance)
        self.likelihood_loss = rr.affine_smooth(self.likelihood_loss, self.X)

    def smooth_objective(self, dual, mode='func', check_feasibility=False):

        dual = self.apply_offset(dual)

        _barrier_star = barrier_conjugate(self.cube_bool,self.inactive_lagrange)

        composition_barrier = rr.affine_smooth(_barrier_star, self.A.T)

        CGF_rand_value, CGF_rand_grad = self.CGF_randomization

        if mode == 'func':
            f_rand_cgf = CGF_rand_value(dual)
            f_data_cgf = self.likelihood_loss.smooth_objective(dual, 'func')
            f_barrier_conj=composition_barrier.smooth_objective(dual, 'func')
            f = self.scale(f_rand_cgf + f_data_cgf + f_barrier_conj-(dual.T.dot(self.dual_arg)))
            #print(f, f_nonneg, f_like, f_active_conj, conjugate_value_i, 'value')
            return f_rand_cgf, f_barrier_conj, f_data_cgf

        elif mode == 'grad':
            g_rand_cgf = CGF_rand_grad(dual)
            g_data_cgf = self.likelihood_loss.smooth_objective(dual, 'grad')
            g_barrier_conj = composition_barrier.smooth_objective(dual, 'grad')
            g = self.scale(g_rand_cgf + g_data_cgf + g_barrier_conj-self.dual_arg)
            # print(g, 'grad')
            return g

        else:
            raise ValueError("mode incorrectly specified")

    #def objective_grad(self,u):
    #    return self.smooth_objective(u, 'grad')

    #def constraint_check(self, check_point):
    #    return self.A_E.dot(check_point)

    #def constraint_ineq(self,u):
    #    return -self.A_E.dot(u)

    def barrier_implicit(self,u):
        if all(self.A_E.dot(u) <= -np.power(10, -20)):
            return np.sum(np.log(1 + np.true_divide(1, -self.A_E.dot(u))))
        return self.A_E.shape[0] * np.log(1 + 10 ** 20)

    def objective(self,u):
        return self.smooth_objective(u,'func') + self.barrier_implicit(u)

    def opt_minimize(self):
        res = minimize(self.objective, x0=self.feasible_point)
        return res.fun, res.x

        #def opt_minimize(self):python test_compare_functions.py
    #    res = fmin_cobyla(self.objective, x0=self.feasible_point, cons=self.constraint_ineq)
    #    return res







