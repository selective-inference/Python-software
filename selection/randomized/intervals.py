from __future__ import print_function, division
import numpy as np

class intervals(object):

    """
    Construct confidence intervals 
    for real-valued parameters by tilting
    a multiparameter exponential family.

    The exponential family is assumed to
    be derived from a Gaussian with
    some selective weight and the
    parameters are linear functionals of the
    mean parameter of the Gaussian.
    
    """
    def __init__(self, reference, sample, observed, covariance):
        '''

        Parameters
        ----------

        reference : np.float(k)
            Reference value of mean parameter. Often
            taken to be an unpenalized MLE or perhaps
            (approximate) selective MLE / MAP.

        sample : np.float(s, k)
            A Monte Carlo sample drawn from selective distribution.

        observed : np.float(k)
            Observed value of Gaussian estimator.
            Often an unpenalized MLE.

        covariance : np.float(k, k)
            Covariance of original Gaussian.
            Used only to compute unselective
            variance of linear functionals of the 
            (approximately) Gaussian estimator.

        '''

        (self.reference,
         self.sample,
         self.observed,
         self.covariance) = (np.asarray(reference),
                            np.asarray(sample),
                            np.asarray(observed),
                            covariance)

        self.shape = reference.shape
        self.nsample = self.sample.shape[1]

    def pivots_all(self, parameter=None):
        '''

        Compute pivotal quantities, i.e.
        the selective distribution function
        under $H_{0,k}:\theta_k=\theta_{0,k}$
        where $\theta_0$ is `parameter`.

        Parameters
        ----------

        parameter : np.float(k) (optional)
            Value of mean parameter under 
            coordinate null hypotheses.
            Defaults to `np.zeros(k)`

        Returns
        -------

        pivots : np.float(k)
            Pivotal quantites. Each is
            (asymptotically) uniformly
            distributed on [0,1] under 
            corresponding $H_{0,k}$.
            
            
        '''
        pivots = np.zeros(self.shape)
        for j in range(self.shape[0]):
            pivots[j] = self._pivot_by_tilting(j, parameter[j])
        return pivots

    def confidence_interval(self, j, alpha=0.1):
        '''

        Construct a `(1-alpha)*100`% confidence
        interval for $\theta_j$ the
        $j$-th coordinate of the mean parameter
        of the underlying Gaussian.

        Parameters
        ----------

        j : int
            Coordinate index in range(self.shape[0])

        alpha : float (optional)
            Specify the (complement of the)
            confidence level.

        Returns
        -------

        L, U : float
            Lower and upper limits of confidence
            interval.
            
        '''
        pvalues_at_grid, grid = self._pvalues_grid(j)
        accepted_indices = np.array(pvalues_at_grid > alpha)
        if np.sum(accepted_indices) > 0:
            L = np.min(grid[accepted_indices])
            U = np.max(grid[accepted_indices])
            return L, U

    def confidence_intervals_all(self, alpha=0.1):
        '''

        Construct a `(1-alpha)*100`% confidence
        interval for each $\theta_j$ 
        of the mean parameter
        of the underlying Gaussian.

        Parameters
        ----------

        alpha : float (optional)
            Specify the (complement of the)
            confidence level.

        Returns
        -------

        LU : np.float(k,2)
            Array with lower and upper confidence limits.
            
        '''

        L, U = np.zeros(self.shape), np.zeros(self.shape)
        for j in range(self.shape[0]):
            LU = self.confidence_interval(j, alpha=alpha)
            if LU is not None:
                L[j], U[j] = LU
        return np.array([L, U]).T

    # Private methods

    def _pivot_by_tilting(self, j, param):
        ref = self.reference[j]
        indicator = np.array(self.sample[:, j] < self.observed[j], dtype =int)
        log_gaussian_tilt = np.array(self.sample[:, j]) * (param - ref)
        log_gaussian_tilt /= self.covariance[j, j]
        emp_exp = self._empirical_exp(j, param)
        LR = np.true_divide(np.exp(log_gaussian_tilt), emp_exp)
        return np.clip(np.sum(np.multiply(indicator, LR)) / float(self.nsample), 0, 1)

    def _pvalues_grid(self, j):
        sd = np.sqrt(self.covariance[j, j])
        grid = np.linspace(-10*sd, 10*sd, 1000) + self.reference[j]
        pvalues_at_grid = [self._pivot_by_tilting(j, grid[i]) 
                           for i in range(grid.shape[0])]
        pvalues_at_grid = [2*min(pval, 1-pval) for pval in pvalues_at_grid]
        pvalues_at_grid = np.asarray(pvalues_at_grid, dtype=np.float32)
        return pvalues_at_grid, grid

    def _empirical_exp(self, j, param):
        ref = self.reference[j]
        factor = (param - ref) / self.covariance[j, j]
        tilted_sample = np.exp(self.sample[:, j] * factor)
        return np.sum(tilted_sample)/float(self.nsample)
