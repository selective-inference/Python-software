import numpy as np

from selection.learning.core import (infer_general_target,
                                  normal_sampler,
                                  logit_fit,
                                  probit_fit)

def simulate(n=100):

    # description of statistical problem

    truth = np.array([2. , -2.]) / np.sqrt(n)

    data = np.random.standard_normal((n, 2)) + np.multiply.outer(np.ones(n), truth) 
    S = np.mean(data, 0)
    observed_sampler = normal_sampler(S, 1/n * np.identity(2))   

    def selection_algorithm(sampler):
        min_success = 1
        ntries = 3
        success = 0
        for _ in range(ntries):
            noisyS = sampler(scale=0.5)
            success += noisyS.sum() > 0.2 / np.sqrt(n)
        return success >= min_success

    # run selection algorithm

    observed_outcome = selection_algorithm(observed_sampler)

    # find the target, based on the observed outcome

    if observed_outcome: # target is truth[0]
        (true_target, 
         observed_target, 
         target_cov, 
         cross_cov) = (truth[0], 
                       S[0], 
                       1./n * np.identity(1), 
                       np.array([1., 0.]).reshape((2,1)) / n)
    else:
        (true_target, 
         observed_target, 
         target_cov, 
         cross_cov) = (truth[1], 
                       S[1], 
                       1./n * np.identity(1), 
                       np.array([0., 1.]).reshape((2,1)) / n)

    pivot, interval = infer_general_target(selection_algorithm,
                                           observed_outcome,
                                           observed_sampler,
                                           observed_target,
                                           cross_cov,
                                           target_cov,
                                           hypothesis=true_target,
                                           fit_probability=probit_fit)[:2]

    return pivot, (interval[0] < true_target) * (interval[1] > true_target), interval[1] - interval[0]

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    n = 100
    U = np.linspace(0, 1, 101)
    P, L = [], []
    plt.clf()
    coverage = 0
    for i in range(300):
        p, cover, l = simulate(n=n)
        coverage += cover
        P.append(p)
        L.append(l)
        print(np.mean(P), np.std(P), np.mean(L) / (2 * 1.65 / np.sqrt(n)), coverage / (i+1))

    plt.clf()
    plt.plot(U, sm.distributions.ECDF(P)(U), 'r', linewidth=3)
    plt.plot([0,1], [0,1], 'k--', linewidth=2)
    plt.show()
