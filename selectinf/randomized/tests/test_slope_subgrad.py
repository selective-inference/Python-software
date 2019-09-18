import numpy as np

from ..slope import _projection_onto_selected_subgradients

def test_projection():

    prox_arg = np.random.normal(0,1,10)
    weights = np.linspace(3, 5, 10)[::-1]
    ordering = np.random.choice(10, 10, replace=False)
    cluster_sizes = [2,3,1,1,3]
    active_signs = np.ones(10)

    proj = _projection_onto_selected_subgradients(prox_arg,
                                                  weights,
                                                  ordering,
                                                  cluster_sizes,
                                                  active_signs)

    print("projection", proj)


