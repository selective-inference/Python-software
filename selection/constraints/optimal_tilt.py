import numpy as np
import regreg.api as rr

class softmax_conjugate(rr.smooth_atom):

    """

    Objective function that computes the value of 

    .. math..

        \inf_{\mu: A\mu \leq b} \frac{1}{2} \|y-z\|^2_2 + \sum_{i=1}^n \log(1 + 1 /(b_i - a_i^T\mu))

    """

    def __init__(self, affine_con, 
                 feasible_point,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                                affine_con.linear_part.shape[1],
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.affine_con = affine_con
        self.feasible_point = feasible_point

    def smooth_objective(self, natural_param, mode='func', check_feasibility=False):

        natural_param = self.apply_offset(natural_param)

        value, minimizer = _solve_softmax_problem(natural_param, 
                                                  self.affine_con,
                                                  self.feasible_point)

        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(minimizer)
        elif mode == 'both':
            return self.scale(value), self.scale(minimizer)
        else:
            raise ValueError('mode incorrectly specified')

class softmax(rr.smooth_atom):

    """
    Softmax function

    .. math..

        \sum_{i=1}^n \log(1 + 1 /(b_i - a_i^T\mu))

    """

    def __init__(self, affine_con, 
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                                affine_con.linear_part.shape[1],
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.affine_con = affine_con

    def smooth_objective(self, natural_param, mode='func', check_feasibility=False):

        natural_param = self.apply_offset(natural_param)
        A = self.affine_con.linear_part
        b = self.affine_con.offset

        slack = b - A.dot(natural_param)
        if np.any(slack < 0):
            raise ValueError('point not feasible')

        value = np.log(1 + 1. / slack).sum()
        grad = -A.T.dot(-1. / slack + 1. / (slack + 1))
        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(grad)
        elif mode == 'both':
            return self.scale(value), self.scale(grad)
        else:
            raise ValueError('mode incorrectly specified')

def _solve_softmax_problem(mean_param, affine_con, feasible_point, niter=40, tol=1.e-8):

    loss = softmax(affine_con)
    A = affine_con.linear_part
    b = affine_con.offset
    coefs = feasible_point.copy()
    step = 1.
    f_cur = np.inf
    for i in range(niter):
        proposed = coefs - step * (coefs - mean_param + loss.smooth_objective(coefs, 'grad'))
        slack = b - A.dot(proposed) 
        if i % 5 == 0:
            step *= 2.
        if np.any(slack < 0):
            step *= 0.5
        else:
            f_proposed = loss.smooth_objective(proposed, 'func') + 0.5 * ((proposed - mean_param)**2).sum()
            if f_proposed > f_cur * (1 + tol):
                step *= 0.5
            else:
                coefs = proposed
                if np.fabs(f_cur - f_proposed) < tol * max([1, f_cur, f_proposed]):
                    break
                f_cur = f_proposed

    return f_proposed, coefs

class optimal_tilt(rr.smooth_atom):

    """
    An objective used to find an
    approximately best tilt for a
    given affine constraint and a given
    direction of interest.

    We approximately solve the problem
    
    ..math::

        \text{min.}_{c,z:A(z + c\eta + \gamma) \leq b} \|z + c \eta\|^2_{\Sigma}

    where the objective is Mahalanobis distance
    for the constraint's covariance, $\gamma$ is
    the constraint's mean and the set
    $\{w:Aw \leq b\}$ is the affine constraint.

    """

    def __init__(self, affine_con, 
                 direction_of_interest,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                                affine_con.linear_part.shape[1] + 1,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.affine_con = affine_con
        self.direction_of_interest = eta = direction_of_interest 

        design = self.design = np.hstack([np.identity(affine_con.dim), 
                                          eta.reshape((-1,1))])

        sqrt_inv = affine_con.covariance_factors()[1]
        Si = np.dot(sqrt_inv.T, sqrt_inv)
        self.Q = np.dot(design.T, np.dot(Si, design))

        gamma = affine_con.mean

        linear_part = np.dot(affine_con.linear_part, design)
        offset = affine_con.offset - np.dot(affine_con.linear_part, 
                                            affine_con.mean)

        scaling = np.sqrt((linear_part**2).sum(1))
        linear_part /= scaling[:,None]
        offset /= scaling

        self.linear_objective = 0.

        smoothing_quadratic = rr.identity_quadratic(1.e-2, 0, 0, 0)
        self.smooth_constraint = rr.nonpositive.affine(linear_part,
                                 -offset).smoothed(
                                 smoothing_quadratic)

    def smooth_objective(self, z, mode='both', check_feasibility=False):

        Qz = np.dot(self.Q, z) 

        if mode == 'both':
            fc, gc = self.smooth_constraint.smooth_objective(z, mode='both')
            g = Qz + self.linear_objective + gc
            f = (z*Qz).sum() * 0.5 + (self.linear_objective*z).sum() + fc
            return f, g
        elif mode == 'grad':
            gc = self.smooth_constraint.smooth_objective(z, mode='grad')
            g = Qz + self.linear_objective + gc
            return g
        elif mode == 'func':
            fc = self.smooth_constraint.smooth_objective(z, mode='func')
            f = (z*Qz).sum() * 0.5 + (self.linear_objective*z).sum() + fc
            return f

    def fit(self, **regreg_args):
        soln = self.soln = self.solve(**regreg_args)
        self.z_soln = soln[:-1]
        self.c_soln = soln[-1]
        self.optimal_point = np.dot(self.design, self.soln)
        self.reweight_func = -self.affine_con.solve(self.optimal_point)
        return self.optimal_point

