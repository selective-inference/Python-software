import numpy as np
import regreg.api as rr
from scipy.optimize import bisect, minimize


def cube_barrier_softmax_coord(z, lam):
    _diff = z - lam
    _sum = z + lam
    if -lam + np.power(10, -10) < z < lam - np.power(10, -10):
        return np.log((_diff - 1.) * (_sum + 1.) / (_diff * _sum))
    else:
        return 2 * np.log(1+(10 ** 10))

class barrier_conjugate_log(rr.smooth_atom):

    """

    Conjugate of a barrier for the 
    product $[0,\infty)^E \times [-\lambda,\lambda]^{-E}$.
    """

    def __init__(self,
                 cube_bool, # -E
                 lagrange, # cube half lengths
                 barrier_scale=None, # maybe scale each coordinate in future?
                 coef=1.,
                 offset=None,
                 quadratic=None):

        p = cube_bool.shape[0]
        orthant_bool = ~cube_bool

        initial = np.ones(p)
        initial[cube_bool] = lagrange * 0.5

        if barrier_scale is None:
            barrier_scale = 1.

        (self.cube_bool,
         self.orthant_bool,
         self.lagrange,
         self.barrier_scale) = (cube_bool,
                                orthant_bool,
                                lagrange,
                                barrier_scale)

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        # here we compute those expressions in the note

        arg = self.apply_offset(arg) # all smooth_objectives should do this....

        cube_arg = arg[self.cube_bool]

        orthant_arg = arg[self.orthant_bool]
        
        if check_feasibility and np.any(orthant_arg >= -tol):
            if mode == 'func':
                return np.inf
            elif mode == 'grad':
                return np.nan * np.ones(self.shape)
            elif mode == 'both':
                return np.inf, np.nan * np.ones(self.shape)
            else:
                raise ValueError('mode incorrectly specified') 

        orthant_maximizer = - 0.5 + np.sqrt(0.25 - 1. / orthant_arg)
        orthant_val = np.sum(orthant_maximizer * orthant_arg -
                             np.log(1 + 1. / orthant_maximizer))

        cube_maximizer = -1. / cube_arg + np.sign(cube_arg) * np.sqrt(1. / cube_arg ** 2 + self.lagrange ** 2)
        cube_val = np.sum(cube_maximizer * cube_arg + np.log(self.lagrange - cube_maximizer) +
                          np.log(self.lagrange + cube_maximizer) - (2 * np.log(self.lagrange)))

        if mode == 'func':
            return cube_val + orthant_val
        elif mode == 'grad':
            g = np.zeros(self.shape)
            g[self.cube_bool] = cube_maximizer
            g[self.orthant_bool] = orthant_maximizer
            return g
        elif mode == 'both':
            g = np.zeros(self.shape)
            g[self.cube_bool] = cube_maximizer
            g[self.orthant_bool] = orthant_maximizer
            return cube_val + orthant_val, g
        else:
            raise ValueError('mode incorrectly specified')


class barrier_conjugate_softmax(rr.smooth_atom):
    """

    Conjugate of a barrier for the
    product $[0,\infty)^E \times [-\lambda,\lambda]^{-E}$.
    """

    def __init__(self,
                 cube_bool,  # -E
                 lagrange,  # cube half lengths
                 barrier_scale=None,  # maybe scale each coordinate in future?
                 coef=1.,
                 offset=None,
                 quadratic=None):

        p = cube_bool.shape[0]
        orthant_bool = ~cube_bool

        initial = np.ones(p)
        self._initial = initial[cube_bool] = lagrange * 0.5

        if barrier_scale is None:
            barrier_scale = 1.

        (self.cube_bool,
         self.orthant_bool,
         self.lagrange,
         self.barrier_scale) = (cube_bool,
                                orthant_bool,
                                lagrange,
                                barrier_scale)

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        # here we compute those expressions in the note

        arg = self.apply_offset(arg)  # all smooth_objectives should do this....

        cube_arg = arg[self.cube_bool]

        orthant_arg = arg[self.orthant_bool]

        if check_feasibility and np.any(orthant_arg >= -tol):
            if mode == 'func':
                return np.inf
            elif mode == 'grad':
                return np.nan * np.ones(self.shape)
            elif mode == 'both':
                return np.inf, np.nan * np.ones(self.shape)
            else:
                raise ValueError('mode incorrectly specified')

        orthant_maximizer = - 0.5 + np.sqrt(0.25 - 1. / orthant_arg)
        orthant_val = np.sum(orthant_maximizer * orthant_arg -
                             np.log(1 + 1. / orthant_maximizer))

        def cube_conjugate_grad(z, u, j):
            _diff = z - self.lagrange[j]  # z - \lambda < 0
            _sum = z + self.lagrange[j]  # z + \lambda > 0
            return u - (1. / (_diff - 1) - 1. / _diff + 1. / (_sum + 1) - 1. / _sum)

        #def cube_conjugate(z, u, j):
        #    return -u * z + cube_barrier_softmax_coord(z, self.lagrange[j])

        cube_maximizer = np.zeros(cube_arg.shape[0])

        #for i in range(cube_arg.shape[0]):
        #    u = cube_arg[i]
        #    j = i
        #    bounds = [(-self.lagrange[i], self.lagrange[i])]
        #    res = minimize(cube_conjugate, x0=(self._initial)[i], args=(u,j), bounds=bounds)
        #    cube_maximizer[i] = res.x

        for i in range(cube_arg.shape[0]):
            u = cube_arg[i]
            j = i
            cube_maximizer[i] = bisect(cube_conjugate_grad, a=-self.lagrange[j] + 10 ** -10,
                                       b=self.lagrange[j] - 10 ** -10, args=(u, j),
                                       rtol=4.4408920985006262e-5, maxiter=32)

        cube_val = np.sum(cube_maximizer * cube_arg - np.log(1. + (1. / self.lagrange - cube_maximizer))
                          - np.log(1. + (1. / self.lagrange + cube_maximizer)))

        if mode == 'func':
            return cube_val + orthant_val
        elif mode == 'grad':
            g = np.zeros(self.shape)
            g[self.cube_bool] = cube_maximizer
            g[self.orthant_bool] = orthant_maximizer
            return g
        elif mode == 'both':
            g = np.zeros(self.shape)
            g[self.cube_bool] = cube_maximizer
            g[self.orthant_bool] = orthant_maximizer
            return cube_val + orthant_val, g
        else:
            raise ValueError('mode incorrectly specified')


class barrier_conjugate_softmax_scaled(rr.smooth_atom):
    """

    Conjugate of a barrier for the
    product $[0,\infty)^E \times [-\lambda,\lambda]^{-E}$.
    """

    def __init__(self,
                 cube_bool,  # -E
                 lagrange,  # cube half lengths
                 barrier_scale=None,  # maybe scale each coordinate in future?
                 coef=1.,
                 offset=None,
                 quadratic=None):

        p = cube_bool.shape[0]
        orthant_bool = ~cube_bool

        initial = np.ones(p)
        self._initial = initial[cube_bool] = lagrange * 0.5

        if barrier_scale is None:
            barrier_scale = 1.

        (self.cube_bool,
         self.orthant_bool,
         self.lagrange,
         self.barrier_scale) = (cube_bool,
                                orthant_bool,
                                lagrange,
                                barrier_scale)

        rr.smooth_atom.__init__(self,
                                (p,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        # here we compute those expressions in the note

        arg = self.apply_offset(arg)  # all smooth_objectives should do this....

        cube_arg = arg[self.cube_bool]

        orthant_arg = arg[self.orthant_bool]

        if check_feasibility and np.any(orthant_arg >= -tol):
            if mode == 'func':
                return np.inf
            elif mode == 'grad':
                return np.nan * np.ones(self.shape)
            elif mode == 'both':
                return np.inf, np.nan * np.ones(self.shape)
            else:
                raise ValueError('mode incorrectly specified')

        orthant_maximizer = - 0.5 + np.sqrt(0.25 - 1. / orthant_arg)
        orthant_val = np.sum(orthant_maximizer * orthant_arg -
                             np.log(1 + 1. / orthant_maximizer))

        def cube_conjugate_grad(z, u, j):
            _diff = z - self.lagrange[j]  # z - \lambda < 0
            _sum = z + self.lagrange[j]  # z + \lambda > 0
            return u - (1. / (_diff - self.lagrange[j]) - 1. / _diff + 1. / (_sum + self.lagrange[j]) - 1. / _sum)

        #def cube_conjugate(z, u, j):
        #    return -u * z + cube_barrier_softmax_coord(z, self.lagrange[j])

        cube_maximizer = np.zeros(cube_arg.shape[0])

        #for i in range(cube_arg.shape[0]):
        #    u = cube_arg[i]
        #    j = i
        #    bounds = [(-self.lagrange[i], self.lagrange[i])]
        #    res = minimize(cube_conjugate, x0=(self._initial)[i], args=(u,j), bounds=bounds)
        #    cube_maximizer[i] = res.x

        for i in range(cube_arg.shape[0]):
            u = cube_arg[i]
            j = i
            cube_maximizer[i] = bisect(cube_conjugate_grad, a=-self.lagrange[j] + 10 ** -10,
                                       b=self.lagrange[j] - 10 ** -10, args=(u, j),
                                       rtol=4.4408920985006262e-5, maxiter=32)

        cube_val = np.sum(cube_maximizer * cube_arg - np.log(1. + (self.lagrange / (self.lagrange - cube_maximizer)))
                          - np.log(1. + (self.lagrange / (self.lagrange + cube_maximizer))))

        if mode == 'func':
            return cube_val + orthant_val
        elif mode == 'grad':
            g = np.zeros(self.shape)
            g[self.cube_bool] = cube_maximizer
            g[self.orthant_bool] = orthant_maximizer
            return g
        elif mode == 'both':
            g = np.zeros(self.shape)
            g[self.cube_bool] = cube_maximizer
            g[self.orthant_bool] = orthant_maximizer
            return cube_val + orthant_val, g
        else:
            raise ValueError('mode incorrectly specified')
