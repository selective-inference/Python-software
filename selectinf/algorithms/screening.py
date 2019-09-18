import numpy as np
from scipy.sparse import eye as sparse_eye

from ..constraints.affine import constraints

def _basis_vector(j,n):
    """
    j-th elementary basis vector in R^n
    """
    e = np.zeros(n)
    e[j] = 1.
    return e

class topK(object):

    alpha = 0.1

    def __init__(self, X, Y, K, sigma, covariance=None):
        n, p = X.shape
        self.Z = np.dot(X.T, Y)
        self.X, self.Y = X, Y
        self.sign = np.sign(self.Z)
        self.covariance = covariance
        self.K = K
        order = np.argsort(np.fabs(self.Z))
        self.selected = order[-K:]
        self.selected_sign = self.sign[order[-K:]]

        partial = np.identity(p)[order[:-K]]
        partial = np.vstack([partial, -partial])

        full_matrix = []
        for k in range(1, K+1):
            partial_cp = partial.copy()
            partial_cp[:,order[-k]] = -self.sign[order[-k]]
            full_matrix.append(np.dot(partial_cp, X.T))
        linear_part = np.vstack(full_matrix)
        self.constraints = constraints(linear_part, 
                                       np.zeros(linear_part.shape[0]),
                                       covariance=covariance)
        self.constraints.covariance *= sigma**2
        self.sigma = sigma

    @property
    def intervals(self, doc="OLS intervals for active variables adjusted for selection."):
        if not hasattr(self, "_intervals"):
            p = self.Z.shape[0]
            self._intervals = []
            C = self.constraints
            for j in self.selected:
                s = self.sign[j]
                eta = self.X[:,j] * s
                _interval = C.interval(eta,
                                       self.Y,
                                       self.alpha)
                self._intervals.append((j, (eta*self.Y).sum(), 
                                        _interval))
        return self._intervals
        
def test():
    n, p, sigma = 40, 100, 1.4
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n) * sigma

    top10 = topK(X, Y, 10, sigma)
    return top10, top10.intervals
