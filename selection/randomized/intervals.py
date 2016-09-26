import numpy as np
from scipy.stats import norm as ndist

class intervals(object):

    def __init__(self):
        self.grid_length = 400
        self.param_values_at_grid = np.linspace(-10, 10, num=self.grid_length)


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
            pval = 2 * min(pval, 1 - pval)
            pvalues.append(pval)
        return pvalues

    def pvalues_ref_all(self):
        pvalues = []
        for j in range(self.nactive):
            indicator = np.array(self.samples[j, :] < self.observed[j], dtype=int)
            pval = np.sum(indicator)/float(self.nsamples)
            pval = 2*min(pval, 1-pval)
            pvalues.append(pval)

        return pvalues


    def pvalues_grid(self, j):
        pvalues_at_grid = [self.pvalue_by_tilting(j, self.param_values_at_grid[i]) 
                           for i in range(self.grid_length)]
        pvalues_at_grid = [2*min(pval, 1-pval) for pval in pvalues_at_grid]
        pvalues_at_grid = np.asarray(pvalues_at_grid, dtype=np.float32)
        return pvalues_at_grid


    def construct_intervals(self, j, alpha=0.1):
        pvalues_at_grid = self.pvalues_grid(j)
        accepted_indices = np.array(pvalues_at_grid > alpha)
        if np.sum(accepted_indices) > 0:
            self.L = np.min(self.param_values_at_grid[accepted_indices])
            self.U = np.max(self.param_values_at_grid[accepted_indices])
            return self.L, self.U

    def construct_intervals_all(self, truth_vec, alpha=0.1):
        ncovered = 0
        nparam = 0
        for j in range(self.nactive):
            LU = self.construct_intervals(j, alpha=alpha)
            if LU is not None:
                L, U = LU
                print "interval", L, U
                nparam += 1
                if (L <= truth_vec[j]) and (U >= truth_vec[j]):
                    ncovered += 1
        return ncovered, nparam

    def construct_naive_intervals(self, truth_vec, alpha=0.1):
        ncovered = 0
        nparam = 0
        quantile = -ndist.ppf(alpha/float(2))
        #print "quantile", quantile
        for j in range(self.nactive):
            sigma = np.sqrt(self.variances[j])
            L = self.ref_vec[j]-sigma*quantile
            U = self.ref_vec[j]+sigma*quantile
            print "naive interval", L, U
            nparam += 1
            if (L <= truth_vec[j]) and (U >= truth_vec[j]):
                ncovered += 1
        return ncovered, nparam
