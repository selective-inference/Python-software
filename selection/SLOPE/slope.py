"""
Implementation of the SLOPE proximal operator of
https://statweb.stanford.edu/~candes/papers/SLOPE.pdf
"""
from copy import copy
import numpy as np
import regreg.api as rr
from scipy import sparse

have_isotonic = False
try:
    from sklearn.isotonic import IsotonicRegression

    have_isotonic = True
except ImportError:
    raise ValueError('unable to import isotonic regression from sklearn')


from regreg.atoms.seminorms import seminorm

from regreg.atoms import _work_out_conjugate
from regreg.objdoctemplates import objective_doc_templater
from regreg.doctemplates import (doc_template_user, doc_template_provider)


@objective_doc_templater()
class slope(seminorm):
    """
    The SLOPE penalty
    """

    objective_template = r"""\sum_j \lambda_j |(var)s_{(j)}|"""

    def __init__(self, weights, lagrange=None, bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):

        weights = np.array(weights, np.float)
        if not np.allclose(-weights, np.sort(-weights)):
            raise ValueError('weights should be non-increasing')
        if not np.all(weights > 0):
            raise ValueError('weights must be positive')

        self.weights = weights
        self._dummy = np.arange(self.weights.shape[0])

        seminorm.__init__(self, self.weights.shape,
                          lagrange=lagrange,
                          bound=bound,
                          quadratic=quadratic,
                          initial=initial,
                          offset=offset)

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, x,
                                     check_feasibility=check_feasibility,
                                     lagrange=lagrange)
        xsort = np.sort(np.fabs(x))[::-1]
        return lagrange * np.fabs(xsort * self.weights).sum()

    @doc_template_user
    def constraint(self, x, bound=None):
        bound = seminorm.constraint(self, x, bound=bound)
        inbox = self.seminorm(x, lagrange=1,
                              check_feasibility=True) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, x, lipschitz, lagrange)
        return _basic_proximal_map(x, self.weights * lagrange / lipschitz)

    @doc_template_user
    def bound_prox(self, x, bound=None):
        raise NotImplementedError

    def __copy__(self):
        return self.__class__(self.weights.copy(),
                              quadratic=self.quadratic,
                              initial=self.coefs,
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange),
                              offset=copy(self.offset))

    def __repr__(self):
        if self.lagrange is not None:
            if not self.quadratic.iszero:
                return "%s(%s, lagrange=%f, offset=%s)" % \
                       (self.__class__.__name__,
                        str(self.weights),
                        self.lagrange,
                        str(self.offset))
            else:
                return "%s(%s, lagrange=%f, offset=%s, quadratic=%s)" % \
                       (self.__class__.__name__,
                        str(self.weights),
                        self.lagrange,
                        str(self.offset),
                        self.quadratic)
        else:
            if not self.quadratic.iszero:
                return "%s(%s, bound=%f, offset=%s)" % \
                       (self.__class__.__name__,
                        str(self.weights),
                        self.bound,
                        str(self.offset))
            else:
                return "%s(%s, bound=%f, offset=%s, quadratic=%s)" % \
                       (self.__class__.__name__,
                        str(self.weights),
                        self.bound,
                        str(self.offset),
                        self.quadratic)

    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            if self.bound is None:
                cls = conjugate_slope_pairs[self.__class__]
                atom = cls(self.weights,
                           bound=self.lagrange,
                           lagrange=None,
                           offset=offset,
                           quadratic=outq)
            else:
                cls = conjugate_slope_pairs[self.__class__]
                atom = cls(self.weights,
                           lagrange=self.bound,
                           bound=None,
                           offset=offset,
                           quadratic=outq)
        else:
            atom = smooth_conjugate(self)

        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    conjugate = property(get_conjugate)


@objective_doc_templater()
class slope_conjugate(slope):
    r"""
    The dual of the slope penalty:math:`\ell_{\infty}` norm
    """

    objective_template = r"""P^*(%(var)s)"""

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, x,
                                     check_feasibility=check_feasibility,
                                     lagrange=lagrange)
        xsort = np.sort(np.fabs(x))[::-1]
        return lagrange * np.fabs(xsort / self.weights).max()

    @doc_template_user
    def constraint(self, x, bound=None):
        bound = seminorm.constraint(self, x, bound=bound)
        inbox = self.seminorm(x, lagrange=1,
                              check_feasibility=True) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        raise NotImplementedError

    @doc_template_user
    def bound_prox(self, x, bound=None):
        bound = seminorm.bound_prox(self, x, bound)

        # the proximal map is evaluated
        # by working out the SLOPE proximal
        # map and computing the residual

        # might be better to just find the correct cython function instead
        # of always constructing IsotonicRegression

        _slope_prox = _basic_proximal_map(x, self.weights * bound)
        return x - _slope_prox


def _basic_proximal_map(center, weights):
    """
    Proximal algorithm described (2.3) of SLOPE
    though sklearn isotonic has ordering reversed.
    """

    # the proximal map sorts the absolute values,
    # runs isotonic regression with an offset
    # reassigns the signs

    # might be better to just find the correct cython function instead
    # of always constructing IsotonicRegression

    ir = IsotonicRegression()

    _dummy = np.arange(center.shape[0])
    _arg = np.argsort(np.fabs(center))
    shifted_center = np.fabs(center)[_arg] - weights[::-1]
    _prox_val = np.clip(ir.fit_transform(_dummy, shifted_center), 0, np.inf)
    _return_val = np.zeros_like(_prox_val)
    _return_val[_arg] = _prox_val
    _return_val *= np.sign(center)
    return _return_val


def _projection_onto_selected_subgradients(prox_arg,
                                           weights,
                                           ordering,
                                           cluster_sizes,
                                           active_signs,
                                           last_value_zero=True):
    """
    Compute the projection of a point onto the set of
    subgradients of the SLOPE penalty with a given
    clustering of the solution and signs of the variables.
    This is a projection onto a lower dimensional set. The dimension
    of this set is p -- the dimensions of the `prox_arg` minus
    the number of unique values in `ordered_clustering` + 1 if the
    last value of the solution was zero (i.e. solution was sparse).
    Parameters
    ----------
    prox_arg : np.ndarray(p, np.float)
        Point to project
    weights : np.ndarray(p, np.float)
        Weights of the SLOPE penalty.
    ordering : np.ndarray(p, np.int)
        Order of original argument to SLOPE prox.
        First entry corresponds to largest argument of SLOPE prox.
    cluster_sizes : sequence
        Sizes of clusters, starting with
        largest in absolute value.
    active_signs : np.ndarray(p, np.int)
         Signs of non-zero coefficients.
    last_value_zero : bool
        Is the last solution value equal to 0?
    """

    result = np.zeros_like(prox_arg)

    ordered_clustering = []
    cur_idx = 0
    for cluster_size in cluster_sizes:
        ordered_clustering.append([ordering[j + cur_idx] for j in range(cluster_size)])
        cur_idx += cluster_size

    # Now, run appropriate SLOPE prox on each cluster
    cur_idx = 0
    for i, cluster in enumerate(ordered_clustering):
        prox_subarg = np.array([prox_arg[j] for j in cluster])

        # If the value of the soln to the prox was non-zero
        # then we solve a SLOPE of size 1 smaller than the cluster

        # If the cluster size is 1, the value is just
        # the corresponding signed weight

        if i < len(ordered_clustering) - 1 or not last_value_zero:
            if len(cluster) == 1:
                result[cluster[0]] = weights[cur_idx] * active_signs[cluster[0]]
            else:
                indices = [j + cur_idx for j in range(len(cluster))]
                cluster_weights = weights[indices]

                ir = IsotonicRegression()
                _ir_result = ir.fit_transform(np.arange(len(cluster)), cluster_weights[::-1])[::-1]
                result[indices] = -np.multiply(active_signs[indices], _ir_result/2.)

        else:
            indices = np.array([j + cur_idx for j in range(len(cluster))])
            cluster_weights = weights[indices]

            pen = slope(cluster_weights, lagrange=1.)
            loss = rr.squared_error(np.identity(len(cluster)), prox_subarg)
            slope_problem = rr.simple_problem(loss, pen)
            result[indices] = prox_subarg - slope_problem.solve()

        cur_idx += len(cluster)

    return result

"""
For a cluster of size bigger than 1, we solve
"""

conjugate_slope_pairs = {}
for n1, n2 in [(slope, slope_conjugate)]:
    conjugate_slope_pairs[n1] = n2
    conjugate_slope_pairs[n2] = n1