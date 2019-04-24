import numpy as np
from selection.learning.core import (infer_full_target,
                                  normal_sampler,
                                  logit_fit,
                                  probit_fit)

def simulate(n=100):

    # description of statistical problem

    truth = np.array([2. , -2.]) / np.sqrt(n)

    dispersion = 2
    data = np.sqrt(dispersion) * np.random.standard_normal((n, 2)) + np.multiply.outer(np.ones(n), truth) 
    S = np.sum(data, 0)
    observed_sampler = normal_sampler(S, dispersion * n * np.identity(2))   

    def selection_algorithm(sampler):
        min_success = 1
        ntries = 3
        success = 0
        for _ in range(ntries):
            noisyS = sampler(scale=0.5)
            success += noisyS.sum() > 0.2 * np.sqrt(n) * np.sqrt(dispersion)
        if success >= min_success:
            return set([1, 0])
        return set([1])

    # run selection algorithm

    observed_set = selection_algorithm(observed_sampler)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths = [], [], []
    for idx in observed_set:
        true_target = truth[idx]

        pivot, interval = infer_full_target(selection_algorithm,
                                            observed_set,
                                            [idx],
                                            observed_sampler,
                                            dispersion,
                                            hypothesis=[true_target],
                                            fit_probability=probit_fit)[0][:2]

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

    return pivots, covered, lengths

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    n = 100
    U = np.linspace(0, 1, 101)
    P, L, coverage = [], [], []
    plt.clf()
    for i in range(300):
        p, cover, l = simulate(n=n)
        coverage.extend(cover)
        P.extend(p)
        L.extend(l)
        print(np.mean(P), np.std(P), np.mean(L) / (2 * 1.65 / np.sqrt(n)), np.mean(coverage))

    plt.clf()
    plt.plot(U, sm.distributions.ECDF(P)(U), 'r', linewidth=3)
    plt.plot([0,1], [0,1], 'k--', linewidth=2)
    plt.show()
