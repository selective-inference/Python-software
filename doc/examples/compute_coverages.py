import numpy as np
import matplotlib.pyplot as plt

from statsmodels.distributions import ECDF
from selection.randomized.tests.test_opt_weighted_intervals import test_opt_weighted_intervals


def compute_coverage(sel_ci, true_vec):
    nactive = true_vec.shape[0]
    coverage = np.zeros(nactive)
    for i in range(nactive):
        if true_vec[i]>=sel_ci[i,0] and true_vec[i]<=sel_ci[i,1]:
            coverage[i]=1
    return coverage


def main(ndraw=20000, burnin=5000, nsim=50):
    np.random.seed(1)

    sel_pivots_all = list()
    sel_ci_all = list()
    rand_all = []
    for i in range(nsim):
        for idx, (rand, sel_pivots, sel_ci, true_vec) in enumerate(test_opt_weighted_intervals(ndraw=ndraw, burnin=burnin)):
            if i==0:
                sel_pivots_all.append([])
                rand_all.append(rand)
                sel_ci_all.append([])
            sel_pivots_all[idx].append(sel_pivots)
            print(sel_ci)
            sel_ci_all[idx].append(compute_coverage(sel_ci, true_vec))

    xval = np.linspace(0, 1, 200)

    for idx in range(len(rand_all)):
        fig = plt.figure(num=idx, figsize=(8,8))
        plt.clf()
        sel_pivots_all[idx] = [item for sublist in sel_pivots_all[idx] for item in sublist]
        plt.plot(xval, ECDF(sel_pivots_all[idx])(xval), label='selective')
        plt.plot(xval, xval, 'k-', lw=1)
        plt.legend(loc='lower right')

        sel_ci_all[idx] = [item for sublist in sel_ci_all[idx] for item in sublist]
        print(sel_ci_all)
        plt.title(''.join(["coverage ", str(np.mean(sel_ci_all[idx]))]))
        plt.savefig(''.join(["fig", rand_all[idx], '.pdf']))

