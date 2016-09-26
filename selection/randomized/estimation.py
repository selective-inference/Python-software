import numpy as np
from scipy.optimize import minimize

class estimation(object):

    def __init__(self, X, y, active, betaE, cube, epsilon, lam, sigma, tau):

        (self.X, self.y,
         self.active,
         self.betaE, self.cube,
         self.epsilon,
         self.lam,
         self.sigma,
         self.tau) = (X, y,
                      active,
                      betaE, cube,
                      epsilon,
                      lam,
                      sigma,
                      tau)

        self.sigma_sq = self.sigma **2

        self.signs = np.sign(self.betaE)
        self.n, self.p = X.shape
        self.nactive = np.sum(active)
        self.ninactive = self.p-self.nactive
        self.XE_pinv = np.linalg.pinv(self.X[:, self.active])

        self.Sigma_inv = [np.array((self.p + 1, self.p + 1)) for _ in range(self.nactive)]
        self.Sigma_full = [np.array((self.p + 1, self.p + 1)) for _ in range(self.nactive)]
        self.Sigma_inv_mu = [np.zeros(self.p + 1) for _ in range(self.nactive)]

        self.eta_norm_sq = np.zeros(self.nactive)
        for j in range(self.nactive):
            eta = self.XE_pinv[j, :]
            self.eta_norm_sq[j] = np.linalg.norm(eta)**2

        self.observed_vec = np.zeros(self.p+1)
        self.observed_vec[1:] = np.concatenate((self.betaE, self.cube), axis=0)


        self.mle = np.zeros(self.nactive)


    def setup_joint_Gaussian_parameters(self, j):
        """
        Sigma_inv_mu computed for beta_{E,j}^*=0
        """
        eta = self.XE_pinv[j, :]

        c = np.true_divide(eta, self.eta_norm_sq[j])
        A = np.zeros((self.p, self.p + 1))
        A[:, 0] = -np.dot(self.X.T, c)
        A[:, 1:(self.nactive + 1)] = np.dot(self.X.T, self.X[:, self.active])
        A[:self.nactive, 1:(self.nactive + 1)] += self.epsilon * np.identity(self.nactive)
        A[self.nactive:, (self.nactive + 1):] = self.lam * np.identity(self.ninactive)
        fixed_part = np.dot(np.identity(self.n) - np.outer(c, eta), self.y)
        gamma = -np.dot(self.X.T, fixed_part)
        gamma[:self.nactive] += self.lam * self.signs

        v = np.zeros(self.p + 1)
        v[0] = 1
        self.Sigma_inv[j] = (np.true_divide(np.dot(A.T, A), self.tau ** 2) +
                             np.true_divide(np.outer(v, v), \
                                 self.eta_norm_sq[j] * (self.sigma ** 2)))
        self.Sigma_full[j] = np.linalg.inv(self.Sigma_inv[j])
        self.Sigma_inv_mu[j] = np.true_divide(np.dot(A.T, gamma), self.tau ** 2)

        return self.Sigma_inv[j], self.Sigma_inv_mu[j]

    def log_selection_probability(self, param, j, method="barrier"):

        # print 'param value', param
        Sigma_inv_mu_modified = self.Sigma_inv_mu[j].copy()
        Sigma_inv_mu_modified[0] += param / (self.eta_norm_sq[j] * (self.sigma ** 2))

        initial_guess = np.zeros(self.p + 1)
        initial_guess[1:(self.nactive + 1)] = self.betaE
        initial_guess[(self.nactive + 1):] = np.random.uniform(-1, 1, self.ninactive)

        bounds = ((None, None),)
        for i in range(self.nactive):
            if self.signs[i] < 0:
                bounds += ((None, 0),)
            else:
                bounds += ((0, None),)
            bounds += ((-1, 1),) * self.ninactive


        def chernoff(x):
            return np.inner(x, self.Sigma_inv[j].dot(x)) / 2 - np.inner(Sigma_inv_mu_modified, x)

        def barrier(x):
            # Ax\leq b
            A = np.zeros((self.p+self.ninactive, 1 + self.p))
            A[:self.nactive, 1:(self.nactive + 1)] = -np.diag(self.signs)
            A[self.nactive:self.p, (self.nactive + 1):] = np.identity(self.ninactive)
            A[self.p:, (self.nactive + 1):] = -np.identity(self.ninactive)
            b = np.zeros(self.p+self.ninactive)
            b[self.nactive:] = 1

            if all(b - np.dot(A, x) >= np.power(10, -9)):
                return np.sum(np.log(1 + np.true_divide(1, b - np.dot(A, x))))

            return b.shape[0] * np.log(1 + 10 ** 9)

        def objective(x):
            return chernoff(x) + barrier(x)

        if method == "barrier":
            res = minimize(objective, x0=initial_guess)
        else:
            if method == "chernoff":
                res = minimize(chernoff, x0=initial_guess, bounds=bounds)
            else:
                raise ValueError('wrong method')

        mu = np.dot(self.Sigma_full[j], Sigma_inv_mu_modified)
        return - np.true_divide(np.inner(mu, Sigma_inv_mu_modified), 2) - res.fun


    def compute_mle(self, j):

        observed_vector = self.observed_vec.copy()
        observed_vector[0] = np.inner(self.XE_pinv[j, :], self.y)

        def objective_mle(param):
            Sigma_inv_mu_modified = self.Sigma_inv_mu[j].copy()
            Sigma_inv_mu_modified[0] += param / (self.eta_norm_sq[j] * (self.sigma ** 2))
            mu = np.dot(self.Sigma_full[j], Sigma_inv_mu_modified)
            return -np.inner(observed_vector, Sigma_inv_mu_modified) + \
                np.true_divide(np.inner(mu, Sigma_inv_mu_modified), 2) + \
                self.log_selection_probability(param, j)

        initial_guess_mle = 0
        res_mle = minimize(objective_mle, x0=initial_guess_mle)
        self.mle[j] = res_mle.x
        return self.mle[j]


    def compute_mle_all(self):

        for j in range(self.nactive):
            self.setup_joint_Gaussian_parameters(j)
            self.compute_mle(j)

        return self.mle

    def mse_mle(self, true_vec):
        return (np.linalg.norm(self.mle-true_vec))**2


class instance(object):

    def __init__(self,
                 n,
                 p,
                 s,
                 snr=5,
                 sigma=1.,
                 rho=0,
                 random_signs=True,
                 scale=True,
                 center=True):
        (self.n,
         self.p,
         self.s,
         self.snr,
         self.sigma,
         self.rho) = (n,
                      p,
                      s,
                      snr,
                      sigma,
                      rho)

        self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
                  np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
        if center:
            self.X -= self.X.mean(0)[None, :]
        if scale:
            self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

        self.beta = np.zeros(p)
        self.beta[:self.s] = self.snr
        if random_signs:
            self.beta[:self.s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
        self.active = np.zeros(p, np.bool)
        self.active[:self.s] = True

    def _noise(self):
        return np.random.standard_normal(self.n)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + self._noise()) * self.sigma
        return self.X, Y, self.beta * self.sigma, np.nonzero(self.active)[0], self.sigma
