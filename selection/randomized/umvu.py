import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import selection
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import test_lasso

from selection.sampling.randomized.intervals.estimation import estimation, instance


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


def MSE_three(snr=5, n=100, p=10, s=0):

    ninstance = 5
    total_mse_mle, total_mse_unbiased, total_mse_umvu = 0, 0, 0
    nvalid_instance = 0
    data_instance = instance(n, p, s, snr)
    tau = 1.
    for i in range(ninstance):
        X, y, true_beta, nonzero, sigma = data_instance.generate_response()
        # print "true param value", true_beta[0]
        random_Z = np.random.standard_normal(p)
        lam, epsilon, active, betaE, cube, initial_soln = selection(X, y, random_Z)

        if lam < 0:
            print "no active covariates"
        else:
            est = umvu(X, y, active, betaE, cube, epsilon, lam, sigma, tau)
            est.compute_unbiased_all()
            true_vec = true_beta[active]

            print "true vector", true_vec
            print "MLE", est.mle, "Unbiased", est.unbiased, "UMVU", est.umvu
            total_mse_mle += est.mse_mle(true_vec)

            mse = est.mse_unbiased(true_vec)
            total_mse_unbiased += mse[0]
            total_mse_umvu += mse[1]
            nvalid_instance +=np.sum(active)

    if nvalid_instance > 0:
        return total_mse_mle/float(nvalid_instance), total_mse_unbiased/float(nvalid_instance), total_mse_umvu/float(nvalid_instance)


def test_estimation_three():
    snr_seq = np.linspace(-10, 10, num=50)
    filter = np.zeros(snr_seq.shape[0], dtype=bool)
    mse_mle_seq, mse_unbiased_seq, mse_umvu_seq = [], [], []

    for i in range(snr_seq.shape[0]):
            print "parameter value", snr_seq[i]
            mse = MSE_three(snr_seq[i])
            if mse is not None:
                mse_mle, mse_unbiased, mse_umvu = mse
                mse_mle_seq.append(mse_mle)
                mse_unbiased_seq.append(mse_unbiased)
                mse_umvu_seq.append(mse_umvu)
                filter[i] = True

    plt.clf()
    plt.title("MSE")
    fig, ax = plt.subplots()
    ax.plot(snr_seq[filter], mse_mle_seq, label = "MLE", linestyle=':', marker='o')
    ax.plot(snr_seq[filter], mse_unbiased_seq, label = "Unbiased")
    ax.plot(snr_seq[filter], mse_umvu_seq, label ="UMVU")

    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    plt.pause(0.01)
    plt.savefig("MSE")


if __name__ == "__main__":
        test_estimation_three()

        while True:
            plt.pause(0.05)
        plt.show()



