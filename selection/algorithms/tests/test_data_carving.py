import numpy as np
from selection.algorithms.lasso import data_carving, instance, data_splitting

def sim():
    X, Y, _, active, sigma = instance()
    print(sigma)
    G = data_carving.gaussian(X, Y, 1., split_frac=0.9, sigma=sigma)
    G.fit()
    if set(active).issubset(G.active) and G.active.shape[0] > len(active):
        return [G.hypothesis_test(G.active[len(active)], burnin=5000, ndraw=10000)]
    return []

def sim2():
    X, Y, _, active, sigma = instance(n=150, s=3)
    G = data_splitting.gaussian(X, Y, 5., split_frac=0.5, sigma=sigma)
    G.fit(use_full=True)
    if set(active).issubset(G.active) and G.active.shape[0] > len(active):
        return [G.hypothesis_test(G.active[len(active)])]
    return []

