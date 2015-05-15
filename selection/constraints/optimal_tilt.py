import numpy as np
import regreg.api as rr

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

