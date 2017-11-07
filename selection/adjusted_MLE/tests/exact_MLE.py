import numpy as np
from scipy.stats import norm as ndist

def grad_CGF(mu, randomization_scale = 0.5, threshold = 2):
    grad = mu + (1. / np.sqrt(1. + randomization_scale ** 2.)) * (ndist.pdf((threshold -mu)
                                                                          / (np.sqrt(1.+randomization_scale ** 2.)))
                                                                / (1.-ndist.cdf(( threshold -mu) /(np.sqrt(1.+randomization_scale ** 2.)))))
    return grad

def fisher_info(mu, randomization_scale = 0.5, threshold = 2):
    hessian = 1.- (1./(1.+ randomization_scale**2.))*(((mu-threshold)/(np.sqrt(1.+randomization_scale**2.)))
                                                    *ndist.pdf((threshold-mu)/(np.sqrt(1.+randomization_scale**2.)))
                                                    / (1.-ndist.cdf((threshold-mu)/(np.sqrt(1.+randomization_scale**2.)))))
    - (1./(1.+randomization_scale**2.))*((ndist.pdf((threshold-mu)/(np.sqrt(1.+randomization_scale**2.)))
                                                     / (1.-ndist.cdf((threshold-mu)/(np.sqrt(1.+randomization_scale**2.)))))**2)

    return hessian


def simulate_truncated(mu, randomization_scale = 0.5, threshold = 2):
    while True:
        Z = np.random.normal(mu, 1, 1)
        W = np.random.normal(0, randomization_scale, 1)
        if (Z + W > threshold):
            return Z


def test_pivot(mu, randomization_scale = 0.5, threshold = 2):
    Z = np.array([simulate_truncated(mu, randomization_scale = randomization_scale, threshold=threshold) for _ in range(25000)])

    mu_seq = np.linspace(-7., 6, num = 2600)
    grad_partition = np.zeros(mu_seq.shape[0])
    for i in range(mu_seq.shape[0]):
        grad_partition[i] = grad_CGF(mu_seq[i])

    pivot = []
    exact_MLE = []
    sd_MLE = 1/ np.sqrt(fisher_info(mu))
    for k in range(Z.shape[0]):
        MLE = mu_seq[np.argmin(np.abs(grad_partition - Z[k]))]
        exact_MLE.append(MLE)
        pivot.append((MLE-mu)/sd_MLE)

    return np.asarray(pivot), np.asarray(exact_MLE)

print(test_pivot(1))