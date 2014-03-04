"""
This module implements the class `truncated_gaussian` which 
performs (conditional) UMPU tests for Gaussians
restricted to a set of intervals.

"""
import numpy as np
from selection.intervals import _CDF
from scipy.stats import norm as ndist, truncnorm 
from scipy.integrate import quad

from mpmath import mp
mp.dps = 30
import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
numpy2ri.activate()

rpy.r("""
    pnorm.interval <- function(mu, ab) {
    ifelse(mean(ab) - mu < 0,
           pnorm(ab[2] - mu) - pnorm(ab[1] - mu),
           pnorm(mu - ab[1]) - pnorm(mu - ab[2]))
}
""")
_Rdnorm = rpy.r("dnorm")
_Rqnorm = rpy.r("qnorm")
_pnorm_interval = rpy.r('pnorm.interval')

def _cdf(a, b, mu=0, use_R=False):
    if not use_R:
        return _cdf2(a-mu,b-mu) # using mpmath
    return np.squeeze(_pnorm_interval(mu, np.array([a,b]))) # using R

def _dnorm(x, use_R=True):
    if not use_R:
        return np.array(mp.npdf(x))
    x = np.asarray(x)
    return np.asarray(_Rdnorm(x)).reshape(x.shape)

def _qnorm(q, use_R=True):
    if not use_R:
        return np.array(mp.erfinv(2*q-1)*mp.sqrt(2))
    q = np.asarray(q)
    return np.asarray(_Rqnorm(q)).reshape(q.shape)

def _cdf2(a, b):
    if a > 0 and b > 0:
        Fa, Fb = mp.ncdf(-a), mp.ncdf(-b)
        return float(Fa - Fb)
    else:
        Fa, Fb = mp.ncdf(a), mp.ncdf(b)
        return float(Fb - Fa)

def truncnorm_cdf(x, a, b):
    """                                                                         
    calculates P(Z < x | a < Z < b) where Z ~ N(0,1)                            
    """
    if a > 0 and b > 0:
        Fx, Fa, Fb = mp.ncdf(-x), mp.ncdf(-a), mp.ncdf(-b)
        return float( ( Fa - Fx ) / ( Fa - Fb ) )
    else:
        Fx, Fa, Fb = mp.ncdf(x), mp.ncdf(a), mp.ncdf(b)
        return float( ( Fx - Fa ) / ( Fb - Fa ) )


class TruncatedGaussianError(ValueError):
    pass

class truncated_gaussian(object):
    
    """
    A Gaussian distribution, truncated to
    """

    use_R = True

    def __init__(self, intervals, mu=0, sigma=1):
        intervals = np.unique(intervals)
        intervals = np.asarray(intervals).reshape(-1)
        # makes assumption intervals are disjoint
        # and the sorted endpoints give the correct
        # set of intervals...
        self._cutoff_array = np.sort(intervals)
        D = self.intervals[:,1]-self.intervals[:,0]
        I = self.intervals[D != 0]
        self._cutoff_array = I.reshape(-1)
        self._mu = mu
        self._sigma = sigma
        self._mu_or_sigma_changed()

    def __array__(self):
        return self.intervals
    
    @property
    def intervals(self):
        return self._cutoff_array.reshape((-1,2))
    
    @property
    def negated(self):
        if not hasattr(self,"_negated"):
            self._negated = truncated_gaussian(np.asarray(-self._cutoff_array[::-1]),
                                               mu=-self.mu,
                                               sigma=self.sigma)
            self._negated.use_R = self.use_R
        return self._negated
    
    # private method to update P and D after a change of parameters

    # WARNING: when switching self.use_R to True after it was False, it is possible P and D
    # will be mpmath values

    def _mu_or_sigma_changed(self):
        mu, sigma = self.mu, self.sigma
        self.P = np.array([_cdf((a-mu)/sigma, 
                                (b-mu)/sigma, use_R=self.use_R) for a, b in self.intervals])
        self.D = np.array([(_dnorm((a-mu)/sigma, use_R=self.use_R), _dnorm((b-mu)/sigma, use_R=self.use_R)) for a, b in self.intervals])

    # mean parameter : mu

    def set_mu(self, mu):
        self._mu = mu
        self._mu_or_sigma_changed()

    def get_mu(self):
        return self._mu

    mu = property(get_mu, set_mu)

    # variance parameter : sigma

    def set_sigma(self, sigma):
        self._sigma = sigma
        self._mu_or_sigma_changed()

    def get_sigma(self):
        return self._sigma

    sigma = property(get_sigma, set_sigma)

    @property
    def delta(self):
        """
        .. math::
 
            \begin{align}
              \delta_\mu(a,b) &\triangleq \int_a^b x\phi(x-\mu)\,dx \\
              &= - \phi(b-\mu) + \phi(a-\mu) +
              \mu\left(\Phi(b-\mu)-\Phi(a-\mu)\right),
            \end{align}

        """
        mu, P, D = self.mu, self.P, self.D
        return D[:,0] - D[:,1] + mu * P
    
    # End of properties

    @staticmethod
    def twosided(thresh, mu=0, sigma=1):
        thresh = np.fabs(thresh)
        return truncated_gaussian([(-np.inf,-thresh),(thresh,np.inf)],
                                  mu=mu, sigma=sigma)
    
    def __repr__(self):
        return '''%s(%s, mu=%0.3e, sigma=%0.3e)''' % (self.__class__.__name__,
                                                      self.intervals,
                                                      self.mu,
                                                      self.sigma)
    
    def CDF(self, observed):
        P, mu, sigma = self.P, self.mu, self.sigma
        z = observed
        k = int(np.floor((self.intervals <= observed).sum() / 2))
        if k < self.intervals.shape[0]:
            if observed > self.intervals[k,0]:
                return (P[:k].sum() + _cdf((self.intervals[k,0] - mu) / sigma, 
                                           (observed - mu)/sigma, use_R=self.use_R)) / P.sum()
            else:
                return P[:k].sum() / P.sum()
        else:
            return 1.

    def quantile(self, q):
        P, mu, sigma = self.P, self.mu, self.sigma
        Psum = P.sum()
        Csum = np.cumsum(np.array([0]+list(P)))
        k = max(np.nonzero(Csum < Psum*q)[0])

        try:
            k = max(np.nonzero(Csum < Psum*q)[0])
        except ValueError:
            if np.isnan(q):
                raise TruncatedGaussianError('invalid quantile')

        pnorm_increment = Psum*q - Csum[k]
        if np.mean(self.intervals[k]) < 0:
            return mu + _qnorm(_cdf(-np.inf,(self.intervals[k,0]-mu)/sigma, use_R=self.use_R) + pnorm_increment, use_R=self.use_R) * sigma
        else:
            return mu - _qnorm(_cdf((self.intervals[k,0]-mu)/sigma, np.inf, use_R=self.use_R) - pnorm_increment, use_R=self.use_R) * sigma
        
    # make a function for vector version?
    def right_endpoint(self, left_endpoint, alpha):
        c1 = left_endpoint # shorthand from Will's code
        mu, P = self.mu, self.P
        alpha1 = self.CDF(left_endpoint)
        if (alpha1 > alpha):
            return np.nan
        alpha2 = np.array(alpha - alpha1)
        return self.quantile(1-alpha2)
            
    def G(self, left_endpoint, alpha):
        """
        $g_{\mu}$ from Will's code
        """
        c1 = left_endpoint # shorthand from Will's code
        mu, P, D = self.mu, self.P, self.D

        const = np.array(1-alpha)*(np.sum(D[:,0]-D[:,1]) + mu*P.sum())
        right_endpoint = self.right_endpoint(left_endpoint, alpha)
        if np.isnan(right_endpoint):
            return np.inf
        valid_intervals = []
        for a, b in self.intervals:
            intersection = (max(left_endpoint, a),
                            min(right_endpoint, b))
            if intersection[1] > intersection[0]:
                valid_intervals.append(intersection)
        if valid_intervals:
            return truncated_gaussian(valid_intervals, mu=self.mu, sigma=self.sigma).delta.sum() - const
        return 0

    def dG(self, left_endpoint, alpha):
        """
        $gg_{\mu}$ from Will's code
        """
        c1 = left_endpoint # shorthand from Will's code
        D = self.D
        return (self.right_endpoint(left_endpoint, alpha) - 
                left_endpoint) * _dnorm((left_endpoint - self.mu) / self.sigma, use_R=self.use_R)
    
    def naive_interval(self, observed, alpha):
        old_mu = self.mu
        lb = self.mu - 20 * self.sigma
        ub = self.mu + 20 * self.sigma
        def F(param):
            self.mu = param
            return self.CDF(observed)
        L = find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        U = find_root(F, 0.5 * alpha, lb, ub)
        self.mu = old_mu
        return np.array([L, U])

    def UMAU_interval(self, observed, alpha,
                      mu_lo=None,
                      mu_hi=None,
                      tol=1.e-8):
        old_mu = self.mu
        try:
            upper = _UMAU(observed,
                          alpha, self,
                          mu_lo=mu_lo,
                          mu_hi=mu_hi,
                          tol=tol)
        except TruncatedGaussianError:
            upper = np.inf

        tg_neg = self.negated
        try:

            lower = -_UMAU(-observed,
                          alpha, tg_neg,
                          mu_lo=mu_hi,
                          mu_hi=mu_lo,
                          tol=tol)

        except:
            lower = -np.inf

        self.mu, self.negated.mu = old_mu, old_mu
        return np.array([lower, upper])

def G(left_endpoints, mus, alpha, tg):
    """
    Compute the $G$ function of `tg(intervals)` over 
    `zip(left_endpoints, mus)`.

    A copy is made of `tg` and its $(\mu,\sigma)$ are not modified.
    """
    tg = truncated_gaussian(tg.intervals)
    results = []
    for left_endpoint, mu in zip(left_endpoints, mus):
        tg.mu = mu
        results.append(tg.G(left_endpoint, alpha))
    return np.array(results)

def dG(left_endpoints, mus, alpha, tg):
    """
    Compute the $G$ function of `tg(intervals)` over 
    `zip(left_endpoints, mus)`.

    A copy is made of `tg` and its $(\mu,\sigma)$ are not modified.
    """
    tg = truncated_gaussian(tg.intervals)
    results = []
    for left_endpoint, mu in zip(left_endpoints, mus):
        tg.mu = mu
        results.append(tg.dG(left_endpoint, alpha))
    return np.array(results)

def _UMAU(observed, alpha, tg, 
         mu_lo=None,
         mu_hi=None,
         tol=1.e-8):

    tg = truncated_gaussian(tg.intervals, sigma=tg.sigma)

    X = observed # shorthand
    if mu_lo is None:
        mu_lo = X
    if mu_hi is None:
        mu_hi = X + 2

    # find upper and lower points for bisection
    tg.mu = mu_lo
    while tg.G(X, alpha) < 0: # mu_too_high
        mu_lo, mu_hi = mu_lo - 2, mu_lo
        tg.mu = mu_lo

    tg.mu = mu_hi
    while tg.G(X, alpha) > 0: # mu_too_low
        mu_lo, mu_hi = mu_hi, mu_hi + 2
        tg.mu = mu_hi

    # bisection
    while mu_hi - mu_lo > tol:
        mu_bar = 0.5 * (mu_lo + mu_hi)
        tg.mu = mu_bar
        if tg.G(X, alpha) < 0:
            mu_hi = mu_bar
        else:
            mu_lo = mu_bar
    return mu_bar

def find_root(f, y, lb, ub, tol=1e-6):
    """
    searches for solution to f(x) = y in (lb, ub), where 
    f is a monotone decreasing function
    """       
    
    # make sure solution is in range
    a, b   = lb, ub
    fa, fb = f(a), f(b)
    
    # assume a < b
    if fa > y and fb > y:
        while fb > y : 
            b, fb = b + (b-a), f(b + (b-a))
    elif fa < y and fb < y:
        while fa < y : 
            a, fa = a - (b-a), f(a - (b-a))
    
    # determine the necessary number of iterations
    max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )

    # bisect (slow but sure) until solution is obtained
    for _ in xrange(max_iter):
        c, fc  = (a+b)/2, f((a+b)/2)
        if fc > y: a = c
        elif fc < y: b = c
    
    return c
