import math
import numpy as np
import warnings

def find_root(f, y, lb, ub, tol=1e-6):
    """
    searches for solution to f(x) = y in (lb, ub), where 
    f is a monotone decreasing function
    
    STOLEN FROM truncated.py...
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


def crit_func(x, leftCut, rightCut):
    C1, gamma1 = leftCut
    C2, gamma2 = rightCut
    return (x < C1) + (x > C2) + gamma1*(x == C1) + gamma2*(x == C2)
        

class DiscreteExFam:
    def __init__(self, suffstat, weights):
        xw = np.array(sorted(zip(suffstat, weights)))
        self._x = xw[:,0]
        self._w = xw[:,1]
        self._w /= self._w.sum() # make sure they are a pmf
        self.n = len(xw)
        self._old_theta = np.nan

    def set_theta(self, _theta):
        if _theta != self._old_theta:
            _thetaX = _theta * self.suffstat
            _largest = _thetaX.max() + 4 # try to avoid overflow, 4 seems arbitrary
            _exp_thetaX = np.exp(_thetaX - _largest)
            _prod = _exp_thetaX * self.weights
            self._partition = np.sum(_prod)
            self._pdf = _prod / self._partition
            self._partition *= np.exp(_largest)
        self._old_theta = _theta

    def get_theta(self):
        return self._old_theta
    theta = property(get_theta, set_theta)

    @property
    def partition(self):
        if hasattr(self, "_partition"):
            return self._partition

    @property
    def suffstat(self):
        return self._x

    @property
    def weights(self):
        return self._w

    def pdf(self, theta):
        self.set_theta(theta) # compute partition if necessary
        return self._pdf
 
    def cdf(self, theta, x=None, gamma=1):
        """
        P(X < x) + gamma * P(X = x)
        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf) - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.suffstat < x)) 
            if x in self.suffstat:
                tr += gamma * pdf[np.where(self.suffstat == x)]
            return tr

    def ccdf(self, theta, x=None, gamma=0, return_unnorm=False):
        """
        complementary cdf:
        P(X > x) + gamma * P(X = x)
        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf[::-1])[::-1] - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.suffstat > x)) 
            if x in self.suffstat:
                tr += gamma * pdf[np.where(self.suffstat == x)]
            return tr

    def E(self, theta, func=lambda x: x):
        """
        Expectation

        Assumes func(x) is vectorized.
        """
        return (func(self.suffstat) * self.pdf(theta)).sum()

    def Var(self, theta, func=lambda x: x):
        """
        Variance
        """
        mu = self.E(theta, func)
        return self.E(theta, lambda x: (func(x)-mu)**2)
        
    def Cov(self, theta, func1, func2):
        """
        Covariance
        """
        mu1 = self.E(theta, func1)
        mu2 = self.E(theta, func2)
        return self.E(theta, lambda x: (func1(x)-mu1)*(func2(x)-mu2))

    def rightCutFromLeft(self, theta, leftCut, alpha=0.05):
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
            cut = self.suffstat[idx]
            pdf_term = np.exp(theta * cut) / self.partition * self.weights[idx]
            ccdf_term = P[idx]
            gamma2 = (alpha2 - ccdf_term) / pdf_term
            return (cut, gamma2)

    def leftCutFromRight(self, theta, rightCut, alpha=0.05):
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
            cut = self.suffstat[idx]
            cdf_term = P[idx]
            pdf_term = np.exp(theta * cut) / self.partition * self.weights[idx]
            gamma1 = (alpha1 - cdf_term) / pdf_term
            return (cut, gamma1)
    
    def critCovFromLeft(self, theta, leftCut, alpha=0.05):
        """
        Covariance of X with phi(X) where phi(X) is the level-alpha test with left cutoff C1, gamma1
        """
        C1, gamma1 = leftCut
        C2, gamma2 = self.rightCutFromLeft(theta, leftCut, alpha)
        if C2 == np.inf:
            return -np.inf
        else:
            return self.Cov(theta, lambda x: x, lambda x: crit_func(x, (C1, gamma1), (C2, gamma2)))

    def critCovFromRight(self, theta, rightCut, alpha=0.05):
        """
        Covariance of X with phi(X) where phi(X) is the level-alpha test with right cutoff C2, gamma2
        """
        C2, gamma2 = rightCut
        C1, gamma1 = self.leftCutFromRight(theta, rightCut, alpha)
        if C1 == -np.inf:
            return np.inf
        else:
            return self.Cov(theta, lambda x: x, lambda x: crit_func(x, (C1, gamma1), (C2, gamma2)))
                                
    def test2Cutoffs(self, theta, alpha=0.05, tol=1e-6):
        """
        Cutoffs of umpu  two-sided test
        """
        C1 = max([x for x in self.suffstat if self.critCovFromLeft(theta, (x, 0), alpha) >= 0])
        gamma1 = find_root(lambda x: self.critCovFromLeft(theta, (C1, x), alpha), 0., 0., 1., tol)
        C2, gamma2 = self.rightCutFromLeft(theta, (C1, gamma1), alpha)
        return (C1,gamma1, C2, gamma2)

    def test2RejectsLeft(self, theta, x, alpha=0.05, auxVar=1.):
        """
        Returns 1 if x in left lobe of umpu two-sided rejection region
        
        We need an auxiliary uniform variable to carry out the randomized test.
        
        Larger auxVar corresponds to "larger" x, so LESS likely to reject
        auxVar = 1 is conservative
        """
        return self.critCovFromLeft(theta, (x, auxVar), alpha) > 0
                
    def test2RejectsRight(self, theta, x, alpha=0.05, auxVar=0.):
        """
        Returns 1 if x in right lobe of umpu two-sided rejection region
        
        We need an auxiliary uniform variable to carry out the randomized test.
        
        Larger auxVar corresponds to x being slightly "larger," so MORE likely to reject.
        auxVar = 0 is conservative.
        """
        return self.critCovFromRight(theta, (x, 1.-auxVar), alpha) < 0

    def test2Rejects(self, theta, x, alpha=0.05, randomize=True, auxVar=None):
        """
        Returns 1 if x in umpu two-sided rejection region
        
        We need an auxiliary uniform variable to carry out the randomized test.
        
        Larger auxVar corresponds to x being slightly "larger." It can be passed in,
            or chosen at random. If randomize=False, we get a conservative test
        """
        if randomize:
            if auxVar is None:
                auxVar = np.random.random()
            rejLeft = self.test2RejectsLeft(theta, x, alpha, auxVar)
            rejRight = self.test2RejectsRight(theta, x, alpha, auxVar)
        else:
            rejLeft = self.test2RejectsLeft(theta, x, alpha)
            rejRight = self.test2RejectsRight(theta, x, alpha)        
        return rejLeft or rejRight
        
    def inter2Upper(self, x, auxVar, alpha=0.05, tol=1e-6):
        """
        upper bound of two-sided umpu interval
        """
        if x < self.suffstat[0] or (x == self.suffstat[0] and auxVar <= alpha):
            return -np.inf # x, auxVar too small, every test rejects left
        if x > self.suffstat[self.n - 2] or (x == self.suffstat[self.n - 2] and auxVar == 1.):
            return np.inf # x, auxVar too large, no test rejects left
        return find_root(lambda theta: -1*self.test2RejectsLeft(theta, x, alpha, auxVar), -0.5, -1., 1., tol)
        
    def inter2Lower(self, x, auxVar, alpha=0.05, tol=1e-6):
        """
        lower bound of two-sided umpu interval
        """
        if x > self.suffstat[self.n-1] or (x == self.suffstat[self.n-1] and auxVar >= 1.-alpha):
            return np.inf # x, auxVar too large, every test rejects right
        if x < self.suffstat[1] or (x == self.suffstat[1] and auxVar == 0.):
            return -np.inf # x, auxVar too small, no test rejects right
        return find_root(lambda theta: 1.*self.test2RejectsRight(theta, x, alpha, auxVar), 0.5, -1., 1., tol)

    def interval(self, x, alpha=0.05, randomize=True, auxVar=None, tol=1e-6):
        """
        umpu two-sided confidence interval
        """
        if randomize:
            if auxVar is None:
                auxVar = np.random.random()
            upper = self.inter2Upper(x, auxVar, alpha, tol)
            lower = self.inter2Lower(x, auxVar, alpha, tol)
        else:
            upper = self.inter2Upper(x, 1., alpha, tol)
            lower = self.inter2Lower(x, 0., alpha, tol)
        return lower, upper
