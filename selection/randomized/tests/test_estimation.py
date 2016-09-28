import numpy as np

from selection.tests.instance import gaussian_instance

def MSE(snr=1, n=100, p=10, s=1):

    ninstance = 1
    total_mse = 0
    nvalid_instance = 0
    data_instance = gaussian_instance(n, p, s, snr)
    tau = 1.
    for i in range(ninstance):
        X, y, true_beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, snr=snr)
        #print "true param value", true_beta[0]
        random_Z = np.random.standard_normal(p)
        lam, epsilon, active, betaE, cube, initial_soln = selection(X, y, random_Z)
        print "active set", np.where(active)[0]
        if lam < 0:
            print "no active covariates"
        else:
            est = estimation(X, y, active, betaE, cube, epsilon, lam, sigma, tau)
            est.compute_mle_all()

            mse_mle = est.mse_mle(true_beta[active])
            print "MLE", est.mle
            total_mse += mse_mle
            nvalid_instance += np.sum(active)

    return np.true_divide(total_mse, nvalid_instance)


def make_a_plot(plot=False):
    snr_seq = np.linspace(-10, 10, num=20)
    mse_seq = []
    for i in range(snr_seq.shape[0]):
        print "parameter value", snr_seq[i]
        mse = MSE(snr_seq[i])
        print "MSE", mse
        mse_seq.append(mse)

    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.title("MSE")
        plt.plot(snr_seq, mse_seq)
        plt.pause(0.01)
        plt.savefig("MSE")

