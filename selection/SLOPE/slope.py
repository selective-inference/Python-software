"""
Projection onto selected subgradients of SLOPE
"""
import numpy as np

have_isotonic = False
try:
    from sklearn.isotonic import IsotonicRegression
    have_isotonic = True
except ImportError:
    raise ValueError('unable to import isotonic regression from sklearn')

from regreg.atoms.slope import _basic_proximal_map

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

            slope_prox = _basic_proximal_map(prox_subarg, cluster_weights)
            result[indices] = prox_subarg - slope_prox

        cur_idx += len(cluster)

    return result

