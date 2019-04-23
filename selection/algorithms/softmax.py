r"""

This module implements the softmax approximation for
a multivariate Gaussian truncated by affine constraints. The approximation
is an approximation of the normalizing constant in the 
likelihood.

"""


from copy import copy

import numpy as np
from regreg.api import smooth_atom

class softmax_objective(smooth_atom):

    r"""
    The softmax objective

    .. math::

         z \mapsto \frac{1}{2} z^TQz + 
         \sum_{i=1}^{m} \log \left(1 + 
         \frac{1}{(b_i-A_i^T z) / s_i} \right)

    Notes
    -----

    Recall Chernoff's approximation for $Z \sim N(0,I_{n \times n})$:

    .. math::

        -\log P_{\mu}(AZ \leq b) \approx \inf_{z:Az \leq b} 
        \frac{1}{2}\|z-\mu\|^2_2
        = \inf_{z} I_K(z) +  \frac{1}{2}\|z-\mu\|^2_2

    where $I_K$ is the constraint for the set $K=\left\{z:Az \leq b \right\}.$

    The softmax approximation is similar to Chernoff's approximation
    though it uses a soft max barrier function 

    .. math::

         \sum_{i=1}^{m}\log\left(1+\frac{1}{b_i-A_i^T z}\right).

    The softmax objective is

    .. math::

         z \mapsto \frac{1}{2} z^TQz + 
         \sum_{i=1}^{m}\log\left(1+\frac{1}{(b_i-A_i^T z) / s_i}\right).

    where $s_i$ are scalings and $Q$ is a precision matrix (i.e. inverse covariance).


    """

    objective_template = r"""\text{softmax}_K\left(%(var)s\right)"""

    def __init__(self, 
                 shape,
                 precision,
                 constraints,
                 feasible_point,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)
        self.precision = precision
        self.coefs[:] = feasible_point

        # not sure we really need to copy
        
        self.constraints = constraints
        con_linear = self.constraints.linear_part
        self.scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    def smooth_objective(self, param, mode='both', check_feasibility=False):
        """

        Evaluate the smooth objective, computing its value, gradient or both.

        Parameters
        ----------

        mean_param : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        param = self.apply_offset(param)

        con = self.constraints # shorthand
        slack = (con.offset - con.linear_part.dot(param)) / self.scaling

        BIG = 1e20
        if mode in ['both', 'func']:
            if np.any(slack < 0):
                f = BIG
            else:
                f = 0.5 * (param * self.precision.dot(param)).sum() + np.log(1. + 1. / slack).sum()
            f = self.scale(f)

        if mode in ['both', 'grad']:
            dslack = (1. / (slack + self.scaling) - 1. / slack)
            g = self.precision.dot(param) - con.linear_part.T.dot(dslack)
            g = self.scale(g)

        if mode == 'both':
            return f, g
        elif mode == 'grad':
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")

