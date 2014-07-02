import numpy as np

from selection.variance_estimation import draw_truncated
from selection.lasso import lasso
from selection.constraints import constraints
from scipy.optimize import bisect
 
from mpmath import mp
mp.dps = 100

n, p, s, sigma = 500, 1000, 50, 10

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
        self.sample = (draw_truncated(self.Y, self.constraint,
                                     ndraw=ndraw,
                                     burnin=burnin)**2).sum(1)

    def cumulants(self, gamma):
        """
        Compute 

        .. math::
           
           \Lambda(\gamma) = \log\left(\int_{\mathbb{R}^n} 
                             e^{\gamma \|z\|^2_2} F(dz)\right)

        as well as its first two derivatives with respect to $\gamma$,
        where $F$ is the empirical distribution of `self.sample`.

        """
        norm_squared = self.sample
        M = norm_squared.mean()
        M0 = float(np.mean([np.exp(gamma*(ns-M)) for ns in norm_squared]))
        M0 *= np.exp(gamma*M)
        M1 = np.mean([np.exp(float(gamma*ns+np.log(ns)-np.log(M0))) for ns in norm_squared])
        M2 = np.mean([np.exp(float(gamma*ns+2*np.log(ns)-np.log(M0))) for ns in norm_squared])
        return M0, M1, (M2-M1**2)

    def newton(self, initial=0, niter=20):
        """
        Match the gamma so that expected norm squared
        matches `(self.Y**2).sum()`.
        """
        gamma = initial
        n2 = (self.Y**2).sum()
        value = np.inf
        for _ in range(niter):
            V, grad, hess = self.cumulants(gamma)
            grad -= n2
            step = 1.
            gamma_trial = gamma - step * grad / hess
            objective_trial = self.cumulants(gamma_trial)[0]
            while objective_trial - gamma_trial*n2 > value:
                step /= 2.
                gamma_trial = gamma - step * grad / hess

            gamma = gamma_trial
            sigma2 = self.sigma**2 / (1 - 2 * self.sigma**2 * gamma)
            sigma = np.sqrt(sigma2)
            print grad, hess, gamma, sigma2
        return sigma

    def bisection(self):
        def f(sigma_hat):
            gamma = 1/(2*self.sigma**2) - 1 /(2*sigma_hat**2)
            v = self.cumulants(gamma)[1] - (self.Y**2).sum()
            print 'here', v, sigma_hat
            print v
        factor = 3
        try:
            return bisect(f, self.sigma/factor, factor*self.sigma, maxiter=20)
        except ValueError:
            factor *= 2
            return bisect(f, self.sigma/factor, factor*self.sigma, maxiter=20)
            
initial = np.linalg.norm(Z) / np.sqrt(Z.shape[0])
V = variance_estimator(Z, Z_constraint, initial=sigma)
V.draw_sample(ndraw=5000)
print V.cumulants(0), (V.Y**2).sum()
print V.cumulants(0.01)
#ghat2 = V.newton()
ghat = V.bisection()
print ghat

