import numpy as np
from scipy.optimize import minimize

def barrier_conjugate_func(cube_bool, lagrange, arg):

    p = cube_bool.shape[0]
    orthant_bool = ~cube_bool

    initial = np.ones(p)
    initial[cube_bool] = lagrange * 0.5

    cube_arg = arg[cube_bool]
    cube_maximizer = -1. / cube_arg + np.sign(cube_arg) * np.sqrt(1. / cube_arg ** 2 + lagrange ** 2)

    orthant_arg = arg[orthant_bool]
    if np.any(orthant_arg >= -1.e-6):
        return np.power(10,10)
    else:
        orthant_maximizer = - 0.5 + np.sqrt(0.25 - 1. / orthant_arg)
        cube_val = np.sum(cube_maximizer * cube_arg + np.log(lagrange - cube_maximizer)
                          + np.log(lagrange + cube_maximizer)-(2*np.log(lagrange)))
        orthant_val = np.sum(orthant_maximizer * orthant_arg - np.log(1 + 1. / orthant_maximizer))
        return cube_val + orthant_val


class dual_selection_probability_func():

    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,
                 noise_variance,
                 rand_variance,
                 epsilon):

        n, p = X.shape
        E = active.sum()

        self.mean_parameter = np.squeeze(mean_parameter)

        self.active = active
        self.noise_variance = noise_variance

        self.rand_variance = rand_variance
        self.inactive_lagrange = lagrange[~active]
        self.active_lagrange = lagrange[active]
        self.feasible_point = feasible_point
        self.active = active

        X_E = self.X_E = X[:, active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([-X[:, active].T, (B_E + epsilon * np.identity(E)) * active_signs[None, :]])
        self.A_inactive = np.hstack([-X[:, ~active].T, (B_mE * active_signs[None, :])])

        self.offset_active = active_signs * lagrange[active]
        self.inactive_subgrad = np.zeros(p - E)

        append = np.zeros((p, p - E))
        append[E:, :] = np.identity(p - E)
        B_p = self.B_p = np.hstack([np.vstack([self.A_active[:, n:], self.A_inactive[:, n:]]), append])
        self.X = X

        self.B_slice = B_p[:E, :]

        self.cube_bool = np.zeros(p, np.bool)
        self.cube_bool[E:] = 1
        self.dual_arg = -np.append(self.offset_active,self.inactive_subgrad)

    def rand_CGF(self,v):
        u = (np.linalg.inv(self.B_p.T)).dot(v)
        return np.true_divide(u.T.dot(u), 2* self.rand_variance)

    def composed_barrier_conjugate(self,v):
        return barrier_conjugate_func(self.cube_bool, self.inactive_lagrange, v)

    def data_CGF(self,v):
        u = (np.linalg.inv(self.B_p.T)).dot(v)
        dev = np.dot(self.X,u)-self.mean_parameter
        return np.true_divide(dev.T.dot(dev),2 * self.noise_variance)

    def dual_objective(self,v):
        return self.rand_CGF(v)+self.data_CGF(v)+ self.composed_barrier_conjugate(v)-(v.T.dot(np.linalg.inv(self.B_p)))\
            .dot(self.dual_arg)

    def minimize_dual(self):
        bounds = []
        for i in range(self.cube_bool.shape[0]):
            if self.opt_vars[i]:
                bounds.append((-np.inf, np.inf))
            else:
                bounds.append((-np.inf, 0))
        res= minimize(self.dual_objective, x0=self.feasible_point)

        return res.fun-np.true_divide(self.mean_parameter.dot(self.mean_parameter), 2 * self.noise_variance), res.x

