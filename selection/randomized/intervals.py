from __future__ import print_function
import numpy as np
from scipy.stats import norm as ndist

class intervals(object):

    def setup_samples(self, ref_vec, samples, observed, variances):
        (self.ref_vec,
         self.samples,
         self.observed,
         self.variances) = (ref_vec,
                            samples,
                            observed,
                            variances)

        self.nactive = ref_vec.shape[0]
        self.nsamples = self.samples.shape[1]

    def empirical_exp(self, j, param):
        ref = self.ref_vec[j]
        factor = np.true_divide(param-ref, self.variances[j])
        tilted_samples = np.exp(self.samples[j, :]*factor)
        return np.sum(tilted_samples)/float(self.nsamples)

    def pvalue_by_tilting(self, j, param):
        ref = self.ref_vec[j]
        indicator = np.array(self.samples[j, :] < self.observed[j], dtype =int)
        log_gaussian_tilt = np.array(self.samples[j, :]) * (param - ref)
        log_gaussian_tilt /= self.variances[j]
        emp_exp = self.empirical_exp(j, param)
        LR = np.true_divide(np.exp(log_gaussian_tilt), emp_exp)
        return np.clip(np.sum(np.multiply(indicator, LR)) / float(self.nsamples), 0, 1)

    def pvalues_param_all(self, param_vec):
        pvalues = []
        for j in range(self.nactive):
            pval = self.pvalue_by_tilting(j, param_vec[j])
            pvalues.append(pval)
        return np.array(pvalues)

    def pvalues_grid(self, j):
        sd = np.sqrt(self.variances[j])
        grid = np.linspace(-10*sd, 10*sd, 1000) + self.ref_vec[j]
        pvalues_at_grid = [self.pvalue_by_tilting(j, grid[i]) 
                           for i in range(grid.shape[0])]
        pvalues_at_grid = [2*min(pval, 1-pval) for pval in pvalues_at_grid]
        pvalues_at_grid = np.asarray(pvalues_at_grid, dtype=np.float32)
        return pvalues_at_grid, grid

    def construct_intervals(self, j, alpha=0.1):
        pvalues_at_grid, grid = self.pvalues_grid(j)
        accepted_indices = np.array(pvalues_at_grid > alpha)
        if np.sum(accepted_indices) > 0:
            L = np.min(grid[accepted_indices])
            U = np.max(grid[accepted_indices])
            return L, U

    def construct_intervals_all(self, alpha=0.1):
        L, U = np.zeros(self.nactive), np.zeros(self.nactive)
        for j in range(self.nactive):
            LU = self.construct_intervals(j, alpha=alpha)
            if LU is not None:
                L[j], U[j] = LU
        return np.array([L, U])

