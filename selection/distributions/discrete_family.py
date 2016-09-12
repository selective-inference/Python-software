"""

This module contains a class for discrete 
1-dimensional exponential families. The main
uses for this class are exact (post-selection)
hypothesis tests and confidence intervals.

"""
import numpy as np
import warnings

from ..truncated.api import find_root

def crit_func(test_statistic, left_cut, right_cut):
    """
    A generic critical function for an interval,
    with weights at the endpoints.

    ((test_statistic < CL) + (test_statistic > CR) + 
     gammaL * (test_statistic == CL) + 
     gammaR * (test_statistic == CR))

    where (CL, gammaL) = left_cut, (CR, gammaR) = right_cut.

    Parameters
    ----------

    test_statistic : np.float
        Observed value of test statistic.

    left_cut : (float, float)
        (CL, gammaL): left endpoint and value at exactly the left endpoint (should be in [0,1]).

    right_cut : (float, float)
        (CR, gammaR): right endpoint and value at exactly the right endpoint (should be in [0,1]).

    Returns
    -------

    decision : np.float

    """
    CL, gammaL = left_cut
    CR, gammaR = right_cut
    value = ((test_statistic < CL) + (test_statistic > CR)) * 1.
    if gammaL != 0:
        value += gammaL * (test_statistic == CL)
    if gammaR != 0:
        value += gammaR * (test_statistic == CR)
    return value

class discrete_family(object):

    def __init__(self, sufficient_stat, weights, theta=0.):
        r"""
        A  discrete 1-dimensional
        exponential family with reference measure $\sum_j w_j \delta_{X_j}$
        and sufficient statistic `sufficient_stat`. For any $\theta$, the distribution
        is

        .. math::
        
            P_{\theta} = \sum_{j} e^{\theta X_j - \Lambda(\theta)} w_j \delta_{X_j}

        where

        .. math::

            \Lambda(\theta) = \log \left(\sum_j w_j e^{\theta X_j} \right).

        Parameters
        ----------

        sufficient_stat : `np.float((n))`

        weights : `np.float(n)`

        Notes
        -----

        The weights are normalized to sum to 1.
        """
        xw = np.array(sorted(zip(sufficient_stat, weights)))
        self._x = xw[:,0]
        self._w = xw[:,1]
        self._lw = np.log(xw[:,1])
        self._w /= self._w.sum() # make sure they are a pmf
        self.n = len(xw)
        self._theta = np.nan
        self.theta = theta

    @property
    def theta(self):
        """
        The natural parameter of the family.
        """
        return self._theta

    @theta.setter
    def theta(self, _theta):
        if _theta != self._theta:
            _thetaX = _theta * self.sufficient_stat + self._lw
            _largest = _thetaX.max() - 5 # try to avoid over/under flow, 5 seems arbitrary
            _exp_thetaX = np.exp(_thetaX - _largest)
            _prod = _exp_thetaX
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
 
    def cdf(self, theta, x=None, gamma=1):
        r"""
        The cumulative distribution function of $P_{\theta}$ with
        weight `gamma` at `x`

        .. math::

            P_{\theta}(X < x) + \gamma * P_{\theta}(X = x)

        Parameters
        ----------

        theta : float
             Natural parameter.

        x : float (optional)
             Where to evaluate CDF.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        cdf : np.float

        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf) - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.sufficient_stat < x)) 
            if x in self.sufficient_stat:
                tr += gamma * np.sum(pdf[np.where(self.sufficient_stat == x)])
            return tr

    def ccdf(self, theta, x=None, gamma=0, return_unnorm=False):
        r"""
        The complementary cumulative distribution function 
        (i.e. survival function) of $P_{\theta}$ with
        weight `gamma` at `x`

        .. math::

            P_{\theta}(X > x) + \gamma * P_{\theta}(X = x)

        Parameters
        ----------

        theta : float
             Natural parameter.

        x : float (optional)
             Where to evaluate CCDF.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        ccdf : np.float

        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf[::-1])[::-1] - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.sufficient_stat > x)) 
            if x in self.sufficient_stat:
                tr += gamma * np.sum(pdf[np.where(self.sufficient_stat == x)])
            return tr

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

    def two_sided_acceptance(self, theta, alpha=0.05, tol=1e-6):
        r"""
        Compute cutoffs of UMPU two-sided test.

        Parameters
        ----------

        theta : float
             Natural parameter.

        alpha : float (optional)
             Size of two-sided test.

        tol : float
             Tolerance for root-finding.

        Returns
        -------

        left_cut : (float, float)
             Boundary and randomization weight for left endpoint.
   
        right_cut : (float, float)
             Boundary and randomization weight for right endpoint.

        """
        if theta != self._theta:
            CL = np.max([x for x in self.sufficient_stat if self._critCovFromLeft(theta, (x, 0), alpha) >= 0])
            gammaL = find_root(lambda x: self._critCovFromLeft(theta, (CL, x), alpha), 0., 0., 1., tol)
            CR, gammaR = self._rightCutFromLeft(theta, (CL, gammaL), alpha)
            self._left_cut, self._right_cut = (CL, gammaL), (CR, gammaR)
        return self._left_cut, self._right_cut

    def two_sided_test(self, theta0, observed, alpha=0.05, randomize=True, auxVar=None):
        r"""
        Perform UMPU two-sided test.

        Parameters
        ----------

        theta0 : float
             Natural parameter under null hypothesis.

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        auxVar : [None, float]
             If randomizing and not None, use this
             as the random uniform variate.

        Returns
        -------

        decision : np.bool
             Is the null hypothesis $H_0:\theta=\theta_0$ rejected?
   
        Notes
        -----

        We need an auxiliary uniform variable to carry out the randomized test.
        Larger auxVar corresponds to x being slightly "larger." It can be passed in,
        or chosen at random. If randomize=False, we get a conservative test.
        """

        if randomize:
            if auxVar is None:
                auxVar = np.random.random()
            rejLeft = self._test2RejectsLeft(theta0, observed, alpha, auxVar)
            rejRight = self._test2RejectsRight(theta0, observed, alpha, auxVar)
        else:
            rejLeft = self._test2RejectsLeft(theta0, observed, alpha)
            rejRight = self._test2RejectsRight(theta0, observed, alpha)        
        return rejLeft or rejRight
        
    def one_sided_test(self, theta0, observed, alternative='greater', alpha=0.05, randomize=True, auxVar=None):
        r"""
        Perform UMPU one-sided test.

        Parameters
        ----------

        theta0 : float
             Natural parameter under null hypothesis.

        observed : float
             Observed sufficient statistic.

        alternative : str
             One of ['greater', 'less']

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        auxVar : [None, float]
             If randomizing and not None, use this
             as the random uniform variate.

        Returns
        -------

        decision : np.bool
             Is the null hypothesis $H_0:\theta=\theta_0$ rejected?
   
        Notes
        -----

        We need an auxiliary uniform variable to carry out the randomized test.
        Larger auxVar corresponds to x being slightly "larger." It can be passed in,
        or chosen at random. If randomize=False, we get a conservative test.
        """

        if alternative not in ['greater', 'less']:
            raise ValueError('alternative must be one of ["greater", "less"]')

        self.theta = theta0
        if randomize:
            if auxVar is None:
                auxVar = np.random.random()
            if alternative == 'greater':
                return self.ccdf(theta0, observed, gamma=auxVar) < alpha
            else:
                return self.cdf(theta0, observed, gamma=auxVar) < alpha
        else:
            if alternative == 'greater':
                return self.ccdf(theta0, observed) < alpha
            else:
                return self.cdf(theta0, observed) < alpha

    def interval(self, observed, alpha=0.05, randomize=True, auxVar=None, tol=1e-6):
        """
        Form UMAU confidence interval.

        Parameters
        ----------

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        auxVar : [None, float]
             If randomizing and not None, use this
             as the random uniform variate.

        Returns
        -------

        lower, upper : float
             Limits of confidence interval.

        """
        if randomize:
            if auxVar is None:
                auxVar = np.random.random()
            upper = self._inter2Upper(observed, auxVar, alpha, tol)
            lower = self._inter2Lower(observed, auxVar, alpha, tol)
        else:
            upper = self._inter2Upper(observed, 1., alpha, tol)
            lower = self._inter2Lower(observed, 0., alpha, tol)
        return lower, upper

    def equal_tailed_interval(self, observed, alpha=0.05, randomize=True, auxVar=None, tol=1e-6):
        """
        Form interval by inverting
        equal-tailed test with $\alpha/2$ in each tail.

        Parameters
        ----------

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        auxVar : [None, float]
             If randomizing and not None, use this
             as the random uniform variate.

        Returns
        -------

        lower, upper : float
             Limits of confidence interval.

        """

        mu = self.E(self.theta, lambda x: x)
        sigma  = np.sqrt(self.Var(self.theta, lambda x: x))
        lb = mu - 20 * sigma
        ub = mu + 20 * sigma
        F = lambda th : self.cdf(th, observed)
        L = find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        U = find_root(F, 0.5 * alpha, lb, ub)
        return L, U

    def equal_tailed_test(self, theta0, observed, alpha=0.05):
        r"""
        Perform UMPU two-sided test.

        Parameters
        ----------

        theta0 : float
             Natural parameter under null hypothesis.

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        auxVar : [None, float]
             If randomizing and not None, use this
             as the random uniform variate.

        Returns
        -------

        decision : np.bool
             Is the null hypothesis $H_0:\theta=\theta_0$ rejected?
   
        Notes
        -----

        We need an auxiliary uniform variable to carry out the randomized test.
        Larger auxVar corresponds to x being slightly "larger." It can be passed in,
        or chosen at random. If randomize=False, we get a conservative test.
        """

        pval = self.cdf(theta0, observed, gamma=0.5)
        return min(pval, 1-pval) < alpha

    def one_sided_acceptance(self, theta, 
                             alpha=0.05, 
                             alternative='greater',
                             tol=1e-6):
        r"""
        Compute the acceptance region cutoffs of UMPU one-sided test.
        
        TODO: Include randomization?

        Parameters
        ----------

        theta : float
             Natural parameter.

        alpha : float (optional)
             Size of two-sided test.

        alternative : str
             One of ['greater', 'less'].

        tol : float
             Tolerance for root-finding.

        Returns
        -------

        left_cut : (float, float)
             Boundary and randomization weight for left endpoint.
   
        right_cut : (float, float)
             Boundary and randomization weight for right endpoint.

        """

        if alternative == 'greater':
            F = self.ccdf(theta, gamma=0.5)
            cutoff = np.min(self.sufficient_stat[F <= alpha])
            acceptance = (-np.inf, cutoff)
        elif alternative == 'less':
            F = self.ccdf(theta, gamma=0.5)
            cutoff = np.max(self.sufficient_stat[F <= alpha])
            acceptance = (cutoff, np.inf)
        else:
            raise ValueError("alternative should be one of ['greater', 'less']")
        return acceptance

    def equal_tailed_acceptance(self, theta0, alpha=0.05):
        r"""
        Compute the acceptance region cutoffs of 
        equal-tailed test (without randomization).
        Therefore, size may not be exactly $\alpha$.

        Parameters
        ----------

        theta0 : float
             Natural parameter under null hypothesis.

        alpha : float (optional)
             Size of two-sided test.

        Returns
        -------

        left_cut : (float, float)
             Boundary and randomization weight for left endpoint.
   
        right_cut : (float, float)
             Boundary and randomization weight for right endpoint.

        """

        F = self.cdf(theta0, gamma=0.5)
        Lcutoff = np.max(self.sufficient_stat[F <= 0.5 * alpha])
        Rcutoff = np.min(self.sufficient_stat[F >= 1 - 0.5*alpha])
        return Lcutoff, Rcutoff

    # Private methods

    def _rightCutFromLeft(self, theta, leftCut, alpha=0.05):
        """
        Given C1, gamma1, choose C2, gamma2 to make E(phi(X)) = alpha
        """
        C1, gamma1 = leftCut
        alpha1 = self.cdf(theta, C1, gamma1)
        if alpha1 >= alpha:
            return (np.inf, 1)
        else:
            alpha2 = alpha - alpha1
            P = self.ccdf(theta, gamma=0)
            idx = np.nonzero(P < alpha2)[0].min()
            cut = self.sufficient_stat[idx]
            pdf_term = np.exp(theta * cut) / self.partition * self.weights[idx]
            ccdf_term = P[idx]
            gamma2 = (alpha2 - ccdf_term) / pdf_term
            return (cut, gamma2)

    def _leftCutFromRight(self, theta, rightCut, alpha=0.05):
        """
        Given C2, gamma2, choose C1, gamma1 to make E(phi(X)) = alpha
        """
        C2, gamma2 = rightCut
        alpha2 = self.ccdf(theta, C2, gamma2)
        if alpha2 >= alpha:
            return (-np.inf, 1)
        else:
            alpha1 = alpha - alpha2
            P = self.cdf(theta, gamma=0)
            idx = np.nonzero(P < alpha1)[0].max()
            cut = self.sufficient_stat[idx]
            cdf_term = P[idx]
            pdf_term = np.exp(theta * cut) / self.partition * self.weights[idx]
            gamma1 = (alpha1 - cdf_term) / pdf_term
            return (cut, gamma1)
    
    def _critCovFromLeft(self, theta, leftCut, alpha=0.05):
        """
        Covariance of X with phi(X) where phi(X) is the level-alpha test with left cutoff C1, gamma1
        """
        C1, gamma1 = leftCut
        C2, gamma2 = self._rightCutFromLeft(theta, leftCut, alpha)
        if C2 == np.inf:
            return -np.inf
        else:
            return self.Cov(theta, lambda x: x, lambda x: crit_func(x, (C1, gamma1), (C2, gamma2)))

    def _critCovFromRight(self, theta, rightCut, alpha=0.05):
        """
        Covariance of X with phi(X) where phi(X) is the level-alpha test with right cutoff C2, gamma2
        """
        C2, gamma2 = rightCut
        C1, gamma1 = self._leftCutFromRight(theta, rightCut, alpha)
        if C1 == -np.inf:
            return np.inf
        else:
            return self.Cov(theta, lambda x: x, lambda x: crit_func(x, (C1, gamma1), (C2, gamma2)))

    def _test2RejectsLeft(self, theta, observed, alpha=0.05, auxVar=1.):
        """
        Returns 1 if x in left lobe of umpu two-sided rejection region
        
        We need an auxiliary uniform variable to carry out the randomized test.
        
        Larger auxVar corresponds to "larger" x, so LESS likely to reject
        auxVar = 1 is conservative
        """
        return self._critCovFromLeft(theta, (observed, auxVar), alpha) > 0
                
    def _test2RejectsRight(self, theta, observed, alpha=0.05, auxVar=0.):
        """
        Returns 1 if x in right lobe of umpu two-sided rejection region
        
        We need an auxiliary uniform variable to carry out the randomized test.
        
        Larger auxVar corresponds to x being slightly "larger," so MORE likely to reject.
        auxVar = 0 is conservative.
        """
        return self._critCovFromRight(theta, (observed, 1.-auxVar), alpha) < 0

    def _inter2Upper(self, observed, auxVar, alpha=0.05, tol=1e-6):
        """
        upper bound of two-sided umpu interval
        """
        if observed < self.sufficient_stat[0] or (observed == self.sufficient_stat[0] and auxVar <= alpha):
            return -np.inf # observed, auxVar too small, every test rejects left
        if observed > self.sufficient_stat[self.n - 2] or (observed == self.sufficient_stat[self.n - 2] and auxVar == 1.):
            return np.inf # observed, auxVar too large, no test rejects left
        return find_root(lambda theta: -1*self._test2RejectsLeft(theta, observed, alpha, auxVar), -0.5, -1., 1., tol)
        
    def _inter2Lower(self, observed, auxVar, alpha=0.05, tol=1e-6):
        """
        lower bound of two-sided umpu interval
        """
        if observed > self.sufficient_stat[self.n-1] or (observed == self.sufficient_stat[self.n-1] and auxVar >= 1.-alpha):
            return np.inf # observed, auxVar too large, every test rejects right
        if observed < self.sufficient_stat[1] or (observed == self.sufficient_stat[1] and auxVar == 0.):
            return -np.inf # observed, auxVar too small, no test rejects right
        return find_root(lambda theta: 1.*self._test2RejectsRight(theta, observed, alpha, auxVar), 0.5, -1., 1., tol)

