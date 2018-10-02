import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from core import (infer_full_target,
                  split_sampler, # split_sampler not working yet
                  normal_sampler,
                  logit_fit,
                  repeat_selection,
                  probit_fit)

def simulate(n=1000, p=50, s=5, signal=(3, 5), sigma=2, alpha=0.1):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(XTX, XTXi, lam, sampler):

        p = XTX.shape[0]
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=XTX)
        pen = rr.l1norm(p, lagrange=lam)

        scale = 0.5
        noisy_S = sampler(scale=scale)
        loss.quadratic = rr.identity_quadratic(0, 0, -noisy_S, 0)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve(max_its=50, tol=1.e-6)
        success += soln != 0
        return set(np.nonzero(success)[0])

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, XTX, XTXi, 4. * np.sqrt(n))

    # run selection algorithm

    success_params = (6, 10)

    observed_set = repeat_selection(selection_algorithm, splitting_sampler, *success_params)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths = [], [], []
    naive_pivots, naive_covered, naive_lengths =  [], [], []

    for idx in observed_set:
        print("variable: ", idx, "total selected: ", len(observed_set))
        true_target = truth[idx]

        (pivot, 
         interval) = infer_full_target(selection_algorithm,
                                       observed_set,
                                       idx,
                                       splitting_sampler,
                                       dispersion,
                                       hypothesis=true_target,
                                       fit_probability=probit_fit,
                                       alpha=alpha,
                                       B=500)

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target-quantile * target_sd, observed_target+quantile * target_sd)
        naive_pivots.append((1-ndist.cdf((observed_target-true_target)/target_sd))) # one-sided

        naive_covered.append((naive_interval[0]<true_target)*(naive_interval[1]>true_target))
        naive_lengths.append(naive_interval[1]-naive_interval[0])

    return pivots, covered, lengths, naive_pivots, naive_covered, naive_lengths


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    np.random.seed(1)

    U = np.linspace(0, 1, 101)
    P, L, coverage = [], [], []
    naive_P, naive_L, naive_coverage = [], [], []
    plt.clf()
    for i in range(30):
        p, cover, l, naive_p, naive_covered, naive_l = simulate()
        coverage.extend(cover)
        P.extend(p)
        L.extend(l)
        naive_P.extend(naive_p)
        naive_coverage.extend(naive_covered)
        naive_L.extend(naive_l)

        print("selective:", np.mean(P), np.std(P), np.mean(L) , np.mean(coverage))
        print("naive:", np.mean(naive_P), np.std(naive_P), np.mean(naive_L), np.mean(naive_coverage))
        print("len ratio selective divided by naive:", np.mean(np.array(L) / np.array(naive_L)))

        if i % 5 == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P)(U), 'r', linewidth=3)
            plt.plot([0,1], [0,1], 'k--', linewidth=2)
            plt.savefig('lasso_example2.pdf')
