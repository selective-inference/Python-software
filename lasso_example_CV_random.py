import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from knockoffs import lasso_glmnet

from core import (infer_full_target,
                  split_sampler, # split_sampler not working yet
                  normal_sampler,
                  logit_fit,
                  probit_fit,
                  repeat_selection)

def simulate(n=100, p=20, s=10, signal=(2, 2), sigma=2, alpha=0.1):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.,
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(X, XTXi, resid, sampler):

        S = sampler(scale=0.5) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        G = lasso_glmnet(X, ynew, *[None]*4)
        select = G.select()
        return set(list(select[0]))

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

    # run selection algorithm

    success_params = (2, 3)

    observed_set = repeat_selection(selection_algorithm, splitting_sampler, *success_params)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths = [], [], []
    naive_pivots, naive_covered, naive_lengths =  [], [], []

    for idx in list(observed_set)[:1]:
        print("variable: ", idx, "total selected: ", len(observed_set))
        true_target = truth[idx]

        (pivot, 
         interval) = infer_full_target(selection_algorithm,
                                       observed_set,
                                       idx,
                                       splitting_sampler,
                                       dispersion,
                                       hypothesis=true_target,
                                       fit_probability=logit_fit,
                                       success_params=success_params,
                                       alpha=alpha,
                                       B=100)

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(dispersion * XTXi[idx, idx])
        observed_target = np.squeeze(XTXi[idx].dot(X.T.dot(y)))
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target-quantile * target_sd, observed_target+quantile * target_sd)
        naive_pivot = (1-ndist.cdf((observed_target-true_target)/target_sd))
        naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
        naive_pivots.append(naive_pivot)

        naive_covered.append((naive_interval[0]<true_target)*(naive_interval[1]>true_target))
        naive_lengths.append(naive_interval[1]-naive_interval[0])

    return pivots, covered, lengths, naive_pivots, naive_covered, naive_lengths


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pickle
    import pandas as pd

    fit_label = "cv_logit"
    seedn = 3
    outfile = "".join([fit_label, str(seedn), ".csv"])
    np.random.seed(seedn)

    U = np.linspace(0, 1, 101)
    P, L, coverage = [], [], []
    naive_P, naive_L, naive_coverage = [], [], []

    plt.clf()
    for i in range(100):
        print("sim", i)
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

        if i % 2 == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P)(U), 'r', label='Selective', linewidth=3)
            plt.plot([0,1], [0,1], 'k--', linewidth=2)
            plt.plot(U, sm.distributions.ECDF(naive_P)(U), 'b', label='Naive', linewidth=3)
            plt.legend()
            plt.savefig('lasso_example_CV_random_logit.pdf')

    df = pd.DataFrame({'coverage': coverage,
                        'pval': P,
                        'length': L,
                        'naive_pval': naive_P,
                        'naive_length': naive_L,
                        'naive_coverage': naive_coverage})

    df.to_csv(outfile)
