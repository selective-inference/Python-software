import numpy as np
from scipy.optimize import minimize

def nonnegative_barrier(z):
    if all(z >= np.power(10, -10)):
        return np.log(1.+(1./z)).sum()
    else:
        return z.shape[0] * np.log(1 + 10 ** 10)

def cube_barrier_log_coord(z, lam):
    _diff = z - lam
    _sum = z + lam
    if -lam + np.power(10, -10) < z < lam - np.power(10, -10):
        return -np.log(_diff)-np.log(_sum)+(2*np.log(lam))
    else:
        return (2 * np.log(10 ** 10))+(2*np.log(lam))

def cube_barrier_softmax_coord(z, lam):
    _diff = z - lam
    _sum = z + lam
    if -lam + np.power(10, -10) < z < lam - np.power(10, -10):
        return np.log((_diff - 1.) * (_sum + 1.) / (_diff * _sum))
    else:
        return 2 * np.log(1+(10 ** 10))

def cube_barrier_softmax(z,lagrange):
    _diff = z - lagrange
    _sum = z + lagrange
    violations = ((_diff >= 0).sum() + (_sum <= 0).sum() > 0)
    if violations == 0:
        return np.log((_diff - 1.) * (_sum + 1.) / (_diff * _sum)).sum()
    else:
        return  z.shape[0] * np.log(1 + (10 ** 10))


class selection_probability_methods():
    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,  # in R^n
                 noise_variance,
                 rand_variance,
                 epsilon):
        n, p = X.shape
        E = active.sum()

        self.mean_parameter=mean_parameter

        self.active = active
        self.noise_variance = noise_variance

        self.rand_variance = rand_variance
        self.inactive_lagrange = lagrange[~active]
        self.initial = np.zeros(n + E, )
        self.initial[n:] = feasible_point
        self.active = active

        self.opt_vars = np.zeros(n + E, bool)
        self.opt_vars[n:] = 1

        X_E = self.X_E = X[:, active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([-X[:, active].T, (B_E + epsilon * np.identity(E)) * active_signs[None, :]])
        self.A_inactive = np.hstack([-X[:, ~active].T, (B_mE * active_signs[None, :])])

        self.offset_active = active_signs * lagrange[active]
        self.inactive_subgrad = np.zeros(p - E)

        self.B_p = np.vstack([self.A_active[:,n:],self.A_inactive[:,n:]])
        self.X = X

        self.cube_bool = np.zeros(p, np.bool)
        self.cube_bool[E:] = 1

    def objective(self,param):

        def cube_problem(arg, method="softmax_barrier"):
            lam = self.inactive_lagrange[0]
            res_seq = []
            if method == "log_barrier":
                def obj_subgrad(u, mu):
                    return (u * mu) + (np.true_divide(u ** 2, 2 * self.rand_variance))+\
                           (np.true_divide(mu ** 2, 2 * self.rand_variance)) + cube_barrier_log_coord(u, lam)

                for i in range(arg.shape[0]):
                    mu = arg[i]
                    res = minimize(obj_subgrad, x0=self.inactive_subgrad[i], args=mu)
                    res_seq.append(res.fun)

                return np.sum(res_seq)

            elif method == "softmax_barrier":
                def obj_subgrad(u, mu):
                    return (u * mu) + (np.true_divide(u ** 2, 2 * self.rand_variance))+\
                           (np.true_divide(mu ** 2, 2 * self.rand_variance)) + cube_barrier_softmax_coord(u, lam)

                for i in range(arg.shape[0]):
                    mu = arg[i]
                    res = minimize(obj_subgrad, x0=self.inactive_subgrad[i], args=mu)
                    res_seq.append(res.fun)

                return np.sum(res_seq)

        if self.active.sum()==1 :
            f_like = np.true_divide(np.linalg.norm((param[~self.opt_vars])[:,None] - self.mean_parameter) ** 2,
                                    2 * self.noise_variance)

        else:
            f_like = np.true_divide(np.linalg.norm(param[~self.opt_vars] - self.mean_parameter) ** 2,
                                    2 * self.noise_variance)

        f_nonneg = nonnegative_barrier(param[self.opt_vars])

        f_active_conj = np.true_divide(np.linalg.norm(self.A_active.dot(param) + self.offset_active)**2,
                                           2 * self.rand_variance)

        conjugate_argument_i = self.A_inactive.dot(param)

        conjugate_value_i = cube_problem(conjugate_argument_i, method="softmax_barrier")

        #constant = np.true_divide(np.dot(conjugate_argument_i.T, conjugate_argument_i), 2)

        return f_nonneg + f_like + f_active_conj + conjugate_value_i

        #return f_nonneg, f_like, f_active_conj, constant, -conjugate_value_i+ constant


    def minimize_scipy(self):

        res = minimize(self.objective, x0=self.initial)

        return res.fun, res.x

    def objective_p(self,param):

        Sigma = np.true_divide(np.identity(self.X.shape[0]), self.noise_variance) + np.true_divide(
            np.dot(self.X, self.X.T), self.rand_variance)

        Sigma_inv = np.linalg.inv(Sigma)

        Sigma_inter = np.true_divide(np.identity(self.X.shape[0]), self.rand_variance) - np.true_divide(np.dot(np.dot(
            self.X.T, Sigma_inv), self.X), self.noise_variance ** 2)

        arg_constant = np.dot(np.true_divide(np.dot(self.B_p.T, self.X.T), self.rand_variance), Sigma_inv)

        linear_coef = np.dot(arg_constant,(np.true_divide(self.mean_parameter, self.noise_variance)
                           + np.true_divide(np.dot(self.X, self.offset_active),self.rand_variance)))\
                    -np.true_divide(np.dot(self.B_p.T,self.offset_active),self.rand_variance)

        quad_coef = np.dot(np.dot(self.B_p.T, Sigma_inter), self.B_p)

        const_coef = np.true_divide(np.dot(np.dot(self.mean_parameter.T, Sigma_inv), self.mean_parameter),
                                    2*(self.noise_variance**2))

        return np.true_divide(np.dot(np.dot(param.T, quad_coef), param), 2)- np.dot(param.T, linear_coef)\
               + nonnegative_barrier(param[~self.cube_bool]) + cube_barrier_softmax(z,self.inactive_lagrange)\
               -const_coef

    def minimize_scipy_p(self):

        initial_guess = np.zeros(self.X.shape[1])
        initial_guess[~self.cube_bool] = self.feasible_point
        res = minimize(self.objective_p, x0=initial_guess)
        return -res.fun, res.x

















































