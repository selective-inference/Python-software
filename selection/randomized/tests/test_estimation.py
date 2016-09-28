from __future__ import print_function
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
        random_Z = np.random.standard_normal(p)
        lam, epsilon, active, betaE, cube, initial_soln = selection(X, y, random_Z) # selection not defined -- is in a file that was deleted
        print("active set", np.where(active)[0])
        if lam < 0:
            print("no active covariates")
        else:
            est = estimation(X, y, active, betaE, cube, epsilon, lam, sigma, tau)
            est.compute_mle_all()

            mse_mle = est.mse_mle(true_beta[active])
            print("MLE", est.mle)
            total_mse += mse_mle
            nvalid_instance += np.sum(active)

    return np.true_divide(total_mse, nvalid_instance)


def MSE_three(snr=5, n=100, p=10, s=0):

    ninstance = 5
    total_mse_mle, total_mse_unbiased, total_mse_umvu = 0, 0, 0
    nvalid_instance = 0
    data_instance = instance(n, p, s, snr)
    tau = 1.
    for i in range(ninstance):
        X, y, true_beta, nonzero, sigma = data_instance.generate_response()
        random_Z = np.random.standard_normal(p)
        lam, epsilon, active, betaE, cube, initial_soln = selection(X, y, random_Z) # selection not defined -- is in a file that was deleted

        if lam < 0:
            print("no active covariates")
        else:
            est = umvu(X, y, active, betaE, cube, epsilon, lam, sigma, tau)
            est.compute_unbiased_all()
            true_vec = true_beta[active]

            print("true vector", true_vec)
            print("MLE", est.mle, "Unbiased", est.unbiased, "UMVU", est.umvu)
            total_mse_mle += est.mse_mle(true_vec)

            mse = est.mse_unbiased(true_vec)
            total_mse_unbiased += mse[0]
            total_mse_umvu += mse[1]
            nvalid_instance +=np.sum(active)

    if nvalid_instance > 0:
        return total_mse_mle/float(nvalid_instance), total_mse_unbiased/float(nvalid_instance), total_mse_umvu/float(nvalid_instance)


def plot_estimation_three():
    snr_seq = np.linspace(-10, 10, num=50)
    filter = np.zeros(snr_seq.shape[0], dtype=bool)
    mse_mle_seq, mse_unbiased_seq, mse_umvu_seq = [], [], []

    for i in range(snr_seq.shape[0]):
            print("parameter value", snr_seq[i])
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


def make_a_plot(plot=False):
    snr_seq = np.linspace(-10, 10, num=20)
    mse_seq = []
    for i in range(snr_seq.shape[0]):
        print("parameter value", snr_seq[i])
        mse = MSE(snr_seq[i])
        print("MSE", mse)
        mse_seq.append(mse)

    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.title("MSE")
        plt.plot(snr_seq, mse_seq)
        plt.pause(0.01)
        plt.savefig("MSE")

