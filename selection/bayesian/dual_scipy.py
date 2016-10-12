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
                 tau,
                 epsilon):

        self.X=X
        self.mean_parameter=mean_parameter
        self.feasible_point = feasible_point
        n, p = X.shape
        E = active.sum()

        self.active = active
        self.noise_variance = noise_variance
        self.tau = tau
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

        self.cube_bool = np.zeros(p, np.bool)
        self.cube_bool[E:] = 1

    def rand_CGF(self,u):
        return np.true_divide(u.T.dot(u),2)

    def composed_barrier_conjugate(self,u):
        return barrier_conjugate_func(self.cube_bool, self.inactive_lagrange, self.A.T.dot(u))

    def data_CGF(self,u):
        dev = self.X.dot(u)-self.mean_parameter
        return np.true_divide(dev.T.dot(dev),2)

    #def barrier_implicit(self,u):
    #    if all(self.A_E.dot(u) <= -0.00000000000001):
    #        return np.sum(np.log(1 + np.true_divide(1, -self.A_E.dot(u))))
    #    return self.A_E.shape[0] * np.log(1 + 10 ** 14)

    def dual_objective(self,u):
        return self.rand_CGF(u)+self.data_CGF(u)+ self.composed_barrier_conjugate(u)-u.T.dot(self.dual_arg)

    def minimize_opt(self):
        res= minimize(self.dual_objective, x0=self.feasible_point)
        return res.fun, res.x

