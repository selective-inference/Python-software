import numpy as np
import regreg.api as rr

def selection(X, y, random_Z, randomization_scale=1, sigma=None, method="theoretical"):
    n, p = X.shape
    loss = rr.glm.gaussian(X,y)
    epsilon = 1. / np.sqrt(n)
    lam_frac = 1.2
    if sigma is None:
        sigma = 1.
    if method == "theoretical":
        lam = 1. * sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p), weights = dict(zip(np.arange(p), W)), lagrange=1.)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, -randomization_scale * random_Z, 0)


    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    active = (initial_soln != 0)
    if np.sum(active) == 0:
        return None
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE = initial_soln[active]
    subgradient = -(initial_grad+epsilon*initial_soln-randomization_scale*random_Z)
    cube = subgradient[~active]/lam
    return lam, epsilon, active, betaE, cube, initial_soln

class instance(object):

    def __init__(self, n, p, s, signal=5, sigma=1., rho=0, random_signs=False, scale =True, center=True):
         (self.n, self.p, self.s,
         self.signal,
         self.sigma,
         self.rho) = (n, p, s,
                      signal,
                      sigma,
                      rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
              np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         self.beta = np.zeros(p)
         self.beta[:self.s] = self.signal
         if random_signs:
             self.beta[:self.s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
         self.active = np.zeros(p, np.bool)
         self.active[:self.s] = True

    def _noise(self):
        return np.random.standard_normal(self.n)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + self._noise()) * self.sigma
        return self.X, Y, self.beta * self.sigma, np.nonzero(self.active)[0], self.sigma


class generate_data_bayesian():

    def __init__(self, n, p, sigma=1., rho=0., scale =True, center=True):
         (self.n, self.p, self.sigma, self.rho) = (n, p, sigma, rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
                   np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         beta_true = np.zeros(p)
         u = np.random.uniform(0.,1.,p)
         for i in range(p):
             if u[i]<= 0.95:
                 beta_true[i] = np.random.laplace(loc=0., scale= 0.05)
             else:
                 beta_true[i] = np.random.laplace(loc=0., scale= 0.5)

         self.beta = beta_true

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X, Y, self.beta * self.sigma, self.sigma

class instance_mixed(object):

    def __init__(self, n, p, s, sigma=1., rho=0, random_signs=False, scale =True, center=True):
         (self.n, self.p, self.s,
         self.sigma,
         self.rho) = (n, p, s,
                     sigma,
                     rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
              np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         self.beta = np.zeros(p)
         self.beta[:self.s] = np.linspace(0.5, 5.0, num=s)
         if random_signs:
             self.beta[:self.s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
         self.active = np.zeros(p, np.bool)
         self.active[:self.s] = True

    def _noise(self):
        return np.random.standard_normal(self.n)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + self._noise()) * self.sigma
        return self.X, Y, self.beta * self.sigma, np.nonzero(self.active)[0], self.sigma