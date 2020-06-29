import numpy as np

from ...tests.instance import gaussian_instance
from ..lasso import lasso, selected_targets


def test_approx_pivot(n=500,
                      p=100,
                      signal_fac=1.,
                      s=5,
                      sigma=2.,
                      rho=0.4,
                      randomizer_scale=1.):

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      s=s,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)
    dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

    W = 1 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

    conv = const(X,
                 Y,
                 W,
                 randomizer_scale=randomizer_scale * dispersion)

    signs = conv.fit()
    nonzero = signs != 0

    if nonzero.sum()>0:
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        grid = np.linspace(- 20., 20., num=401)

        approximate_grid_inf = conv.approximate_grid_inference(observed_target,
                                                               cov_target,
                                                               cov_target_score,
                                                               grid=grid,
                                                               dispersion=dispersion)

        pivot = approximate_grid_inf.approx_pivot(beta_target)

        return pivot


import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def main(nsim=300):

    _pivot=[]
    for i in range(nsim):

        _pivot.extend(test_approx_pivot(n=100,
                                        p=50,
                                        signal_fac=0.5,
                                        s=5,
                                        sigma=2.,
                                        rho=0.20,
                                        randomizer_scale=1.))

        print("iteration completed ", i)

    plt.clf()
    ecdf_MLE = ECDF(np.asarray(_pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

if __name__ =="__main__":
    main(nsim=50)