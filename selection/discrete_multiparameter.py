"""

This module contains a class for discrete 
multiparameter exponential families. The main
use for this is (post-selection)
maximum likelihood estimation.

Unlike the single parameter family, 
we do not provide tools for exact
hypothesis tests and selection intervals.

Approximate (Wald) tests and intervals are 
possible after estimation.

"""
import numpy as np
import warnings

class multiparameter_family(object):

    def __init__(self, sufficient_stat, weights):
        r"""
        A  discrete multi-parameter dimensional
        exponential family with reference measure $\sum_j w_j \delta_{X_j}$
        and sufficient statistic `sufficient_stat`. 
        For any $\theta$, the distribution is

        .. math::
        
            P_{\theta} = \sum_{j} e^{\theta X_j - \Lambda(\theta)} w_j \delta_{X_j}

        where

        .. math::

            \Lambda(\theta) = \log \left(\sum_j w_j e^{\theta X_j} \right).

        Parameters
        ----------

        sufficient_stat : `np.float((n,k))`

        weights : `np.float(n)`

        Notes
        -----

        The weights are normalized to sum to 1.
        """

        sufficient_stat = np.asarray(sufficient_stat)
        self.n, self.k = sufficient_stat.shape

        weights = np.asarray(weights)
        if weights.shape != (self.n,):
            raise ValueError('expecting weights to have same number of rows as sufficient_stat')

        self._w = weights / weights.sum()
        self._x = np.asarray(sufficient_stat)
        self._theta = np.ones(self.k) * np.nan

    @property
    def theta(self):
        """
        The natural parameter of the family.
        """
        return self._theta

    @theta.setter
    def theta(self, _theta):
        _theta = np.asarray(_theta)
        if not np.all(np.equal(_theta, self._theta)):
            _thetaX = np.dot(self.sufficient_stat, _theta)
            _largest = _thetaX.max() + 4 # try to avoid over/under flow, 4 seems arbitrary
            _exp_thetaX = np.exp(_thetaX - _largest)
            _prod = _exp_thetaX * self.weights
            self._partition = np.sum(_prod)
            self._pdf = _prod / self._partition
            self._partition *= np.exp(_largest)
        self._theta = _theta

    @property
    def partition(self):
        r"""
        Partition function at `self.theta`:

        .. math::

            \sum_j e^{\theta X_j} w_j
        """
        if hasattr(self, "_partition"):
            return self._partition

    @property
    def sufficient_stat(self):
        """
        Sufficient statistics of the exponential family.
        """
        return self._x

    @property
    def weights(self):
        """
        Weights of the exponential family.
        """
        return self._w

    def pdf(self, theta):
        r"""
        Density of $P_{\theta}$ with respect to $P_0$.

        Parameters
        ----------

        theta : float
             Natural parameter.

        Returns
        -------

        pdf : np.float
        
        """
        self.theta = theta # compute partition if necessary
        return self._pdf

    def E(self, theta, func):
        r"""
        Expectation of `func` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func : callable
             Assumed to be vectorized.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        E : np.float

        """
        return (func(self.sufficient_stat) * self.pdf(theta)).sum()

    def Var(self, theta, func):
        r"""
        Variance of `func` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func : callable
             Assumed to be vectorized.

        Returns
        -------

        var : np.float

        """

        mu = self.E(theta, func)
        return self.E(theta, lambda x: (func(x)-mu)**2)
        
    def Cov(self, theta, func1, func2):
        r"""
        Covariance of `func1` and `func2` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func1, func2 : callable
             Assumed to be vectorized.

        Returns
        -------

        cov : np.float

        """

        mu1 = self.E(theta, func1)
        mu2 = self.E(theta, func2)
        return self.E(theta, lambda x: (func1(x)-mu1)*(func2(x)-mu2))

    def mean(self, theta):
        r"""

        Mean parameter of family at natural
        parameter `theta`.

        Parameters
        ----------

        theta : np.float(k)
             Natural parameter.

        Returns
        -------

        mean : np.float(k)
             Expected value of sufficient statistic at theta.

        """
        pdf = self.pdf(theta)
        return (self.sufficient_stat * pdf[:,None]).sum(0)

    def information(self, theta):
        r"""

        Compute mean and Fisher information 
        of family at natural parameter `theta`.

        Parameters
        ----------

        theta : np.float(k)
             Natural parameter.

        Returns
        -------

        mean : np.float(k)
             Expected value of sufficient statistic at theta.

        information : np.float((k,k))
             Covariance matrix of sufficient statistic at theta.
        """
        pdf = self.pdf(theta)
        mean = self.mean(theta)
        outer_prods = np.einsum('ij,ik->ijk', self.sufficient_stat, self.sufficient_stat)
        information = (outer_prods * pdf[:,None,None]).sum(0) - np.outer(mean, mean)
        return mean, information

    def MLE(self, observed_sufficient,
            min_iters=3,
            max_iters=20, 
            tol=1.e-6,
            initial=None):
        r"""

        Compute mean and Fisher information 
        of family at natural parameter `theta`.

        Parameters
        ----------

        observed_sufficient : np.float(k)
            Observed sufficient statistic.

        min_iters : int
            Minimum number of Newton steps.

        max_iters : int
            How many Newton steps?

        tol : float
            Relative tolerance for objective value.

        Returns
        -------

        theta_hat : np.float(k)
            Maximum likelihood estimate.

        """

        _old_theta, _old_pdf, _old_partition = (self._theta.copy(), 
                                                self._pdf.copy(), 
                                                self._partition)

        if initial is None:
            if np.all(np.isnan(self.theta)):
                initial = np.zeros(k)
            else:
                initial = self.theta
        theta = initial
        value = np.inf

        for i in range(max_iters):
            mean, hess = self.information(theta)
            grad = mean - observed_sufficient

            direction = np.linalg.solve(hess, grad)
            step_size = 1.
            while True:
                proposed_theta =  theta - step_size * direction
                self.theta = proposed_theta
                proposed_value = np.log(self.partition) - (self.theta * observed_sufficient).sum()

                if proposed_value < value:
                    break
                else:
                    step_size *= 0.9

            # when do we stop?
            if i > min_iters and np.fabs((value - proposed_value) / value) < tol:
                break
 
            value = proposed_value
            theta = proposed_theta

        # reset old values -- should we?

        self._theta, self._pdf, self._partition = (_old_theta,
                                                   _old_pdf,
                                                   _old_partition)

        return theta
