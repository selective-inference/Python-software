"""
This module contains a class for
forming confindence intervals and
testing 1-dimensional linear hypotheses
about the underlying mean vector of
a Gaussian subjected to selection.
"""

from __future__ import print_function, division
import numpy as np

class intervals_from_sample(object):

    """
    Construct confidence intervals
    for real-valued parameters by tilting
    a multiparameter exponential family
    with reference measure a Monte Carlo sample.

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
            linear_func = np.zeros(self.shape)
            linear_func[j] = 1.
            pivots[j] = self._pivot_param(linear_func, parameter[j])
        return pivots

    def confidence_interval(self, linear_func, level=0.9):
        '''

        Construct a `level*100`% confidence
        interval for a linear functional
        of the mean parameter
        of the underlying Gaussian.

        Parameters
        ----------

        linear_func : np.float(k)
            Linear functional determining
            parameter.

        level : float (optional)
            Specify the
            confidence level.

        Returns
        -------

        L, U : float
            Lower and upper limits of confidence
            interval.
        '''
        alpha = 1 - level
        pvalues_at_grid, grid = self._pivots_grid(linear_func)
        accepted_indices = np.array(pvalues_at_grid > alpha)
        if np.sum(accepted_indices) > 0:
            lower = np.min(grid[accepted_indices])
            upper = np.max(grid[accepted_indices])
            return lower, upper

    def confidence_intervals_all(self, level=0.9):
        '''
        Construct a `level*100`% confidence
        interval for each $\theta_j$
        of the mean parameter
        of the underlying Gaussian.

        Parameters
        ----------

        level : float (optional)
            Specify the confidence level.

        Returns
        -------

        LU : np.float(k,2)
            Array with lower and upper confidence limits.
        '''

        lower, upper = np.zeros(self.shape), np.zeros(self.shape)
        for j in range(self.shape[0]):
            linear_func = np.zeros(self.shape)
            linear_func[j] = 1.
            limits = self.confidence_interval(linear_func, level=level)
            if limits is not None:
                lower[j], upper[j] = limits
            else:
                lower[j], upper[j] = np.nan, np.nan # bad reference -- all pvalues less then alpha
        return np.array([lower, upper]).T

    # Private methods

    def _pivot_param(self, linear_func, param):
        """
        Compute pivotal quantity for the
        quantitiy linear_func.dot(parameter)
        at the hypothesized value param.
        """
        linear_func = np.atleast_1d(linear_func)
        ref = (linear_func * self.reference).sum()
        var = np.sum(linear_func * self.covariance.dot(linear_func))

        _sample = self.sample.dot(linear_func)
        _observed = (self.observed * linear_func).sum()

        indicator = _sample < _observed
        log_gaussian_tilt = _sample  * (param - ref)
        log_gaussian_tilt /= var
        emp_exp = self._empirical_exp(linear_func, param)
        likelihood_ratio = np.exp(log_gaussian_tilt) / emp_exp

        if emp_exp>0:
            return np.clip(np.mean(indicator * likelihood_ratio), 0, 1)
        else:
            return 0.

    def _pivots_grid(self, linear_func, npts=1000, num_sd=10):
        """
        Compute pivots on a 1D grid centered at
        (reference*linear_func).sum() and reference.
        """
        linear_func = np.atleast_1d(linear_func)
        stdev = np.sqrt(np.sum(linear_func * self.covariance.dot(linear_func)))

        #grid = np.linspace(-300*stdev, 300*stdev, 30000) + (self.reference * linear_func).sum()
        grid = np.linspace(-50, 50, 10000) #+ (self.reference * linear_func).sum()
        #print(self.observed.dot(linear_func), 300*stdev)

        pivots_at_grid = [self._pivot_param(linear_func, grid[i])
                          for i in range(grid.shape[0])]
        pivots_at_grid = [2*min(pval, 1-pval) for pval in pivots_at_grid]
        pivots_at_grid = np.asarray(pivots_at_grid)
        return pivots_at_grid, grid

    def _empirical_exp(self, linear_func, param):
        """
        Empirical expected value of the exponential.
        """
        linear_func = np.atleast_1d(linear_func)
        ref = (self.reference * linear_func).sum()
        var = np.sum(linear_func * self.covariance.dot(linear_func))
        factor = (param - ref) / var

        # we can probably save a little bit of time
        # by caching _sample
        _sample = self.sample.dot(linear_func)

        tilted_sample = np.exp(_sample * factor)

        return tilted_sample.mean()
