import numpy as np
import regreg.api as rr

class barrier_conjugate(rr.smooth_atom):

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
        cube_maximizer = -1. / cube_arg + np.sign(cube_arg) * np.sqrt(1. / cube_arg**2 + self.lagrange**2)

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

        if mode in ['func', 'both']:
            cube_val = np.sum(cube_maximizer * cube_arg + 
                              np.log(self.lagrange - cube_maximizer) + 
                              np.log(self.lagrange + cube_maximizer))
            orthant_val = np.sum(orthant_maximizer * orthant_arg - 
                                 np.log(1 + 1. / orthant_maximizer))

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
