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
        initial = np.zeros(n + E, )
        initial[n:] = feasible_point
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
                    res_seq.append(-res.fun)

                return np.sum(res_seq)

            elif method == "softmax_barrier":
                def obj_subgrad(u, mu):
                    return (u * mu) + (np.true_divide(u ** 2, 2 * self.rand_variance))+\
                           (np.true_divide(mu ** 2, 2 * self.rand_variance)) + cube_barrier_softmax_coord(u, lam)

                for i in range(arg.shape[0]):
                    mu = arg[i]
                    res = minimize(obj_subgrad, x0=self.inactive_subgrad[i], args=mu)
                    res_seq.append(-res.fun)

                return np.sum(res_seq)


        f_like = np.true_divide(np.linalg.norm(param[~self.opt_vars]-self.mean_parameter)**2,
                                     2 * self.noise_variance)

        f_nonneg = nonnegative_barrier(param[self.opt_vars])

        f_active_conj = np.true_divide(np.linalg.norm(self.A_active.dot(param) + self.offset_active)**2,
                                           2 * self.rand_variance)

        conjugate_argument_i = self.A_inactive.dot(param)

        conjugate_value_i = cube_problem(conjugate_argument_i, method="softmax_barrier")

        constant = np.true_divide(np.dot(conjugate_argument_i.T, conjugate_argument_i), 2)

        #return f_nonneg + f_like + f_active_conj + constant, -conjugate_value_i + constant

        return f_nonneg + f_like + f_active_conj + constant, conjugate_value_i+ constant









































