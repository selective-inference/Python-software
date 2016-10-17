"""

This module implements the softmax approximation for
a multivariate Gaussian truncated by affine constraints. The approximation
is an approximation of the normalizing constant in the 
likelihood.

Recall Chernoff's approximation for $Z \sim N(0,I_{n \times n})$:
$$
-\log P(AZ \leq b) \approx \inf_{\mu:A\mu \leq b} \frac{1}{2}\|\mu\|^2_2
= \inf_{\mu} I_K(\mu) +  \frac{1}{2}\|\mu\|^2_2
$$
where $I_K(\mu)$ is the constraint for the set $K=\left\{\mu:A\mu \leq b\right\}.$

The softmax approximation is similar to Chernoff's approximation
though it uses a soft max barrier function 
$$ + \sum_{i=1}^{m}\log\left(1+\frac{1}{b_i-A_i^T \mu}\right)$$
where $A_{m \times k}$ with $i$-th row $A_i$ normalized to have length 1.

The softmax approximation solves
$$
\text{minimize}_{\mu} \frac{1}{2} \|\mu\|^2_2 + \sum_{i=1}^{m}\log\left(1+\frac{1}{b_i-A_i^T \mu}\right).
$$

"""

from copy import copy

import numpy as np
from regreg.api import smooth_atom

class softmax_objective(smooth_atom):

    """
    The softmax objective

    .. math::

         \mu \mapsto \frac{1}{2} \|\mu\|^2_2 + 
         \sum_{i=1}^{m} \log \left(1 + 
         \frac{1}{b_i-A_i^T \mu} \right)

    """

    objective_template = r"""\text{softmax}_K\left(%(var)s\right)"""

    def __init__(self, 
                 shape,
                 whitened_constraints,
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

        self.coefs[:] = feasible_point

        # not sure we really need to copy
        
        self.whitened_constraints = copy(whitened_constraints)
        con = whitened_constraints
        _scale = np.sqrt((con.linear_part**2).sum(1))
        con.linear_part /= _scale[:, None]
        con.offset /= _scale

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
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
        
        mean_param = self.apply_offset(mean_param)

        con = self.whitened_constraints # shorthand
        slack = con.offset - con.linear_part.dot(mean_param)

        if mode in ['both', 'func']:
            f = (mean_param**2).sum() + np.log((slack + 1.) / slack).sum()
            f = self.scale(f)

        if mode in ['both', 'grad']:
            dslack = 1. / (slack + 1.) - 1. / slack
            g = mean_param - con.linear_part.T.dot(dslack)
            g = self.scale(g)

        if mode == 'both':
            return f, g
        elif mode == 'grad':
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian(self, mean_param):
        """
        Hessian of the loss.

        Parameters
        ----------

        mean_param : ndarray
            Parameters where Hessian will be evaluated.

        Returns
        -------

        hess : ndarray
            A 1D-array representing the diagonal of the Hessian
            evaluated at `mean_param`.
        """
        mean_param = self.apply_offset(mean_param)

        # shorthand
        con = self.whitened_constraints
        L = con.linear_part

        slack = con.offset - L.dot(mean_param)
        ddslack = 1. / slack**2 - 1. / (slack + 1.)**2
        return L.T.dot(L * ddslack[:, None])

    def get_data(self):
        return self.response

    def set_data(self, data):
        self.response = data

    data = property(get_data, set_data)

    # End loss API


class nonnegative_softmax(smooth_atom):

    """
    The nonnegative softmax objective

    .. math::

         \mu \mapsto
         \sum_{i=1}^{m} \log \left(1 + 
         \frac{1}{\mu_i} \right)

    """

    objective_template = r"""\text{nonneg_softmax}\left(%(var)s\right)"""

    def __init__(self, 
                 shape,
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

        # a feasible point
        self.coefs[:] = np.ones(shape)

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
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
        
        slack = self.apply_offset(mean_param)

        if mode in ['both', 'func']:
            if np.all(slack > 0):
                f = self.scale(np.log((slack + 1.) / slack).sum())
            else:
                f = np.inf
        if mode in ['both', 'grad']:
            g = self.scale(1. / (slack + 1.) - 1. / slack)

        if mode == 'both':
            return f, g
        elif mode == 'grad':
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")

