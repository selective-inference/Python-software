import numpy as np
from scipy.optimize import minimize, bisect

def cube_barrier_softmax_coord(z, lam):
    _diff = z - lam
    _sum = z + lam
    if -lam + np.power(10, -10) < z < lam - np.power(10, -10):
        return np.log((_diff - lam) * (_sum + lam) / (_diff * _sum))
    else:
        return 2 * np.log(1+(10 ** 10))

def softmax_barrier_conjugate(cube_bool, lagrange, arg):
    p = cube_bool.shape[0]
    orthant_bool = ~cube_bool

    initial = np.ones(p)
    initial[cube_bool] = lagrange * 0.5

    cube_arg = arg[cube_bool]

    def cube_conjugate(z,u,j):
        return -u*z + cube_barrier_softmax_coord(z,lagrange[j])

    def cube_conjugate_grad(z,u,j):
        _diff = z - lagrange[j]  # z - \lambda < 0
        _sum = z + lagrange[j]  # z + \lambda > 0
        return u -(1. / (_diff - lagrange[j]) - 1. / _diff + 1. / (_sum + lagrange[j]) - 1. / _sum)

    #cube_val = 0
    #for i in range(cube_arg.shape[0]):
    #    u = cube_arg[i]
    #    j = i
    #    bounds = [(-lagrange[i], lagrange[i])]
    #    res = minimize(cube_conjugate, x0=(initial[cube_bool])[i], args=(u,j), bounds=bounds)
    #    cube_val+= -res.fun

    cube_maximizer = np.zeros(cube_arg.shape[0])
    for i in range(cube_arg.shape[0]):
        u = cube_arg[i]
        j = i
        cube_maximizer[i]= bisect(cube_conjugate_grad, a= -lagrange[j]+10**-10, b= lagrange[j]-10**-10, args=(u,j),
                                  rtol=4.4408920985006262e-5, maxiter=32)

    cube_val = np.sum(cube_maximizer * cube_arg - np.log(1.+(lagrange/(lagrange - cube_maximizer)))
                      - np.log(1.+(lagrange/(lagrange + cube_maximizer))))

    orthant_arg = arg[orthant_bool]

    if np.any(orthant_arg >= -1.e-6):
        return np.power(10, 10)
    else:
        orthant_maximizer = - 0.5 + np.sqrt(0.25 - 1. / orthant_arg)
        orthant_val = np.sum(orthant_maximizer * orthant_arg - np.log(1 + 1. / orthant_maximizer))
        return cube_val + orthant_val


def log_barrier_conjugate(cube_bool, lagrange, arg):

    p = cube_bool.shape[0]
    orthant_bool = ~cube_bool
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

        self._X = X

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

        self.B_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :], np.zeros((E, p - E))])
        self.B_inactive = np.hstack([B_mE * active_signs[None, :], np.identity((p - E))])
        self.B_p = np.vstack((self.B_active, self.B_inactive))

        self.offset_active = active_signs * lagrange[active]
        self.inactive_subgrad = np.zeros(p - E)

        self.offset_active = active_signs * lagrange[active]
        self.inactive_subgrad = np.zeros(p - E)

        self.cube_bool = np.zeros(p, np.bool)
        self.cube_bool[E:] = 1
        self.dual_arg = np.linalg.inv(self.B_p).dot(np.append(self.offset_active, self.inactive_subgrad))


    def rand_CGF(self,v):
        u = (np.linalg.inv(self.B_p.T)).dot(v)
        return np.true_divide(u.T.dot(u), 2* self.rand_variance)

    def composed_barrier_conjugate(self,v):
        return softmax_barrier_conjugate(self.cube_bool, self.inactive_lagrange, v)

    def data_CGF(self,v):
        u = (np.linalg.inv(self.B_p.T)).dot(v)
        dev = np.dot(self._X,u)-self.mean_parameter
        return np.true_divide(dev.T.dot(dev),2 * self.noise_variance)

    def linear_term(self,v):
        return v.T.dot(self.dual_arg)

    def dual_objective(self,v):
        return self.rand_CGF(v)+self.data_CGF(v)+ self.composed_barrier_conjugate(v)+ self.linear_term(v)\
               -np.true_divide(self.mean_parameter.dot(self.mean_parameter), 2 * self.noise_variance)

    def minimize_dual(self):
        bounds = []
        for i in range(self.cube_bool.shape[0]):
            if self.cube_bool[i]:
                bounds.append((-np.inf, np.inf))
            else:
                bounds.append((-np.inf, 0))
        res= minimize(self.dual_objective, x0 = self.feasible_point, bounds = bounds)

        #print(res.fun, self.dual_objective(res.x))
        return res.fun-np.true_divide(self.mean_parameter.dot(self.mean_parameter), 2 * self.noise_variance), res.x


