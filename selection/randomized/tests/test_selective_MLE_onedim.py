import numpy as np
import nose.tools as nt

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim
from selection.tests.instance import gaussian_instance
import matplotlib.pyplot as plt
from selection.randomized.selective_MLE import solve_UMVU, solve_barrier_nonneg


def test_onedim_lasso(n=200, p=1, signal_fac=1.5, signal=1., s=1, ndraw=5000, burnin=1000, sigma=1., full=True, rho=0.4, randomizer_scale=1.):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, highdim.gaussian
    #signal = signal_fac * np.sqrt(2 * np.log(p+1.))

    # X, Y, beta = inst(n=n,
    #                   p=p,
    #                   signal=signal,
    #                   s=s,
    #                   equicorrelated=False,
    #                   rho=rho,
    #                   sigma=sigma,
    #                   random_signs=True)[:3]
    while True:
        X = 1. / np.sqrt(n) * np.ones((n, 1))
        beta = np.zeros(p)
        signal = np.atleast_1d(signal)
        if signal.shape == (1,):
            beta[:s] = signal[0]
        else:
            beta[:s] = np.linspace(signal[0], signal[1], s)
        #beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
        #np.random.shuffle(beta)

        Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

        n, p = X.shape

        W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p + 1.)) * sigma

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma,
                     ridge_term=0.)

        signs = conv.fit()
        # print("conjugate_arg from test", (1./9.)*(signs*np.sqrt(n)*np.mean(Y) - W))
        print("target lin and target offset from test", signs, -W)
        nonzero = signs != 0
        if nonzero.sum()>0:
            break

    target_Z = np.sqrt(n) * np.mean(Y)
    target_transform = (-np.identity(1), np.zeros(1))
    s = signs
    opt_transform = (s * np.identity(1), (s * W) * np.ones(1))
    approx_MLE = solve_UMVU(target_transform,
                            opt_transform,
                            target_Z,
                            np.ones(1),
                            (sigma ** 2.) * np.identity(1),
                            (1. / (sigma ** 2.)) * np.identity(1))

    estimate, _, _, pv = conv.selective_MLE(target="full")
    print(estimate, approx_MLE, 'selective MLE')
    print(beta[nonzero], 'truth')
    print(np.linalg.pinv(X[:, nonzero]).dot(Y), 'relaxed')
    print(pv)

    return estimate, approx_MLE

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fac_seq = np.linspace(-6., 6., 100)
    MLE_now = []
    MLE_prev = []
    for i in range(100):
        test = test_onedim_lasso(n=200, p=1, signal_fac=1.5, signal= fac_seq[i], s=1, ndraw=5000, burnin=1000,
                                 sigma=1., full=True, rho=0.4,randomizer_scale=1.)

        MLE_now.append(test[0])
        MLE_prev.append(test[1])

    plt.plot(fac_seq, np.array(MLE_now), label='MLE now')
    plt.plot(fac_seq, np.array(MLE_prev), 'r--', label='MLE prev')
    plt.legend()
    plt.show()

def main(nsim=500):

    P0, PA = [], []
    from statsmodels.distributions import ECDF

    n, p = 500, 200

    for i in range(nsim):
        try:
            p0, pA = test_highdim_lasso(n=n, p=p, full=True)
        except:
            p0, pA = [], []
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(P0), np.std(P0), np.mean(np.array(PA) < 0.05))

        if i % 3 == 0 and i > 0:
            U = np.linspace(0, 1, 101)
            plt.clf()
            if len(P0) > 0:
                plt.plot(U, ECDF(P0)(U))
            if len(PA) > 0:
                plt.plot(U, ECDF(PA)(U), 'r')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.savefig("plot.pdf")
    plt.show()

