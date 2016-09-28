from __future__ import print_function
import numpy as np
from scipy.optimize import minimize

from selection.randomized.estimation import estimation, instance

class umvu(estimation):

    def __init__(self, X, y, active, betaE, cube, epsilon, lam, sigma, tau):
        estimation.__init__(self, X, y, active, betaE, cube, epsilon, lam, sigma, tau)
        estimation.compute_mle_all(self)
        self.unbiased = np.zeros(self.nactive)
        self.umvu = np.zeros(self.nactive)

    def log_selection_probability_umvu(self, mu, Sigma, method="barrier"):

        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_inv_mu = np.dot(Sigma_inv, mu)

        initial_guess = np.zeros(self.p)
        initial_guess[:self.nactive] = self.betaE
        initial_guess[self.nactive:] = np.random.uniform(-1, 1, self.ninactive)

        bounds = ((None, None),)
        for i in range(self.nactive):
            if self.signs[i] < 0:
                bounds += ((None, 0),)
            else:
                bounds += ((0, None),)
            bounds += ((-1, 1),) * self.ninactive

        def chernoff(x):
            return np.inner(x, Sigma_inv.dot(x)) / 2 - np.inner(Sigma_inv_mu, x)

        def barrier(x):
            # Ax\leq b
            A = np.zeros((self.p + self.ninactive, self.p))
            A[:self.nactive,:self.nactive] = -np.diag(self.signs)
            A[self.nactive:self.p, self.nactive:] = np.identity(self.ninactive)
            A[self.p:, self.nactive:] = -np.identity(self.ninactive)
            b = np.zeros(self.p + self.ninactive)
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

        return res.x


    def compute_unbiased(self, j):

        Sigma22_inv_Sigma21 = np.dot(np.linalg.inv(self.Sigma_full[j][1:, 1:]), self.Sigma_full[j][0, 1:])

        schur = self.Sigma_full[j][0, 0] - np.inner(self.Sigma_full[j][0, 1:], Sigma22_inv_Sigma21)
        c = np.true_divide(self.sigma_sq * self.eta_norm_sq[j], schur)
        a = self.sigma_sq * self.eta_norm_sq[j] * self.Sigma_inv_mu[j][0]

        observed_vector = self.observed_vec.copy()
        observed_vector[0] = np.inner(self.XE_pinv[j, :], self.y)

        self.unbiased[j] = c * (observed_vector[0] - np.inner(Sigma22_inv_Sigma21, observed_vector[1:])) - a

        # starting umvu
        Sigma_tilde = self.Sigma_full[j][1:, 1:]- np.true_divide(np.outer(self.Sigma_full[j][0, 1:], self.Sigma_full[j][0, 1:]), self.Sigma_full[j][0, 0])
        mu_tilde = np.dot(Sigma_tilde.copy(), self.Sigma_inv_mu[j][1:])
        mu_tilde += self.Sigma_full[j][0,1:]*observed_vector[0]/self.Sigma_full[j][0,0]
        z_star = self.log_selection_probability_umvu(mu_tilde.copy(), Sigma_tilde.copy())

        self.umvu[j] = c * (observed_vector[0] - np.inner(Sigma22_inv_Sigma21, z_star)) - a
        return self.unbiased[j], self.umvu[j]


    def compute_unbiased_all(self):
        for j in range(self.nactive):
            self.compute_unbiased(j)
        return self.unbiased, self.umvu

    def mse_unbiased(self, true_vec):
        return (np.linalg.norm(self.unbiased-true_vec))**2, (np.linalg.norm(self.umvu-true_vec))**2


