import numpy as np

from selection.variance_estimation import draw_truncated
from selection.lasso import lasso
from selection.constraints import constraints
from scipy.optimize import bisect
 
from mpmath import mp
mp.dps = 40

n, p, s, sigma = 100, 200, 10, 10

X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,None]
X -= X.mean(0)[None,:]
X /= X.std(0)[None,:]
X /= np.sqrt(n)

beta = np.zeros(p)
beta[:s] = 1.5 * np.sqrt(2 * np.log(p)) * sigma
y = np.random.standard_normal(n) * sigma
L = lasso(y, X, frac=0.5)
L.fit(tol=1.e-14, min_its=200)
C = L.inactive_constraints

PR = np.identity(n) - L.PA
try:
    U, D, V = np.linalg.svd(PR)
except np.linalg.LinAlgError:
    D, U = np.linalg.eigh(PR)

keep = D >= 0.5
U = U[:,keep]
Z = np.dot(U.T, y)
Z_inequality = np.dot(C.inequality, U)
Z_constraint = constraints((Z_inequality, C.inequality_offset), None)

class variance_estimator(object):

    def __init__(self, Y, con, initial=1):
        self.Y = Y
        self.constraint = constraints((con.inequality,
                                       con.inequality_offset),
                                      None)
        self.sigma = initial
        self.constraint.covariance = self.sigma**2 * np.identity(con.dim)
        
    def draw_sample(self, ndraw=2000, burnin=1000):
        """
        Draw a sample from the truncated normal
        storing it as the attribute `sample`.
        """
        if hasattr(self, "sample"):
            state = self.sample[-1].copy()
        else:
            state = self.Y.copy()
        self.sample = draw_truncated(self.Y, self.constraint,
                                     ndraw=ndraw,
                                     burnin=burnin)

    def objective(self, gamma):
        """
        Compute 

        .. math::
           
           \Lambda(\gamma) = \log\left(\int_{\mathbb{R}^n} 
                             e^{\gamma \|z\|^2_2} F(dz)\right)

        as well as its first two derivatives with respect to $\gamma$,
        where $F$ is the empirical distribution of `self.sample`.

        """
        norm_squared = (self.sample**2).sum(1)
        if gamma > 0:
            M = norm_squared.max()
            M0 = np.mean([mp.exp(gamma*(ns-M)) for ns in norm_squared])
            M1 = np.mean([ns*mp.exp(gamma*(ns-M)) for ns in norm_squared])
            M2 = np.mean([(ns**2)*mp.exp(gamma*(ns-M)) for ns in norm_squared])
        else:
            M = norm_squared.min()
            M0 = np.mean([mp.exp(gamma*(ns-M)) for ns in norm_squared])
            M1 = np.mean([ns*mp.exp(gamma*(ns-M)) for ns in norm_squared])
            M2 = np.mean([(ns**2)*mp.exp(gamma*(ns-M)) for ns in norm_squared])
            
        return M0*mp.exp(gamma*M), M1/M0, (M2-M1**2)/M0

    def newton(self, initial=0, niter=10):
        """
        Match the gamma so that expected norm squared
        matches `(self.Y**2).sum()`.
        """
        gamma = initial
        for _ in range(niter):
            V, grad, hess = self.objective(gamma)
            grad -= (self.Y**2).sum()
            gamma = gamma - grad / hess
            print grad, hess
        sigma = np.sqrt(-1 / (2 * (gamma - 1/(2*self.sigma**2))))
        return sigma

    def bisection(self):
        def f(sigma_hat):
            print 'here', sigma_hat
            gamma = 1/(2*V.sigma**2) - 1 /(2*sigma_hat**2)
            return V.objective(gamma)[1] - (V.Y**2).sum()
        factor = 10
        try:
            return bisect(f, self.sigma, factor*self.sigma, maxiter=5)
        except ValueError:
            factor *= 5
            return bisect(f, self.sigma, factor*self.sigma, maxiter=5)
            

initial = np.linalg.norm(Z) / np.sqrt(Z.shape[0])
V = variance_estimator(Z, Z_constraint, initial=initial)
V.draw_sample(ndraw=5000)
print V.objective(0.01)
ghat = V.bisection()
print ghat

