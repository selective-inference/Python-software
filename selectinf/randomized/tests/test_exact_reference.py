import numpy as np

from ...tests.instance import gaussian_instance
from ..lasso import lasso, selected_targets
from ..exact_reference import exact_grid_inference

def test_approx_pivot(n=500,
                      p=100,
                      signal_fac=1.,
                      s=5,
                      sigma=2.,
                      rho=0.4,
                      randomizer_scale=1.):

    while True:

        inst, const = gaussian_instance, lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=True,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        eps = np.random.standard_normal((n, 2000)) * Y.std()
        W = 0.7 * np.median(np.abs(X.T.dot(eps)).max(1))

        conv = const(X,
                     Y,
                     W,
                     ridge_term=0.)
                     #randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs = conv.fit()
        nonzero = signs != 0
        print("size of selected set ", nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

            exact_grid_inf = exact_grid_inference(conv,
                                                  observed_target,
                                                  cov_target,
                                                  cov_target_score)

            pivot = exact_grid_inf._pivots(beta_target)

            return pivot

def main(nsim=300):

    import matplotlib as mpl
    mpl.use('tkagg')
    import matplotlib.pyplot as plt
    from statsmodels.distributions.empirical_distribution import ECDF

    _pivot = []
    for i in range(nsim):
        _pivot.extend(test_approx_pivot(n=400,
                                        p=100,
                                        signal_fac=0.5,
                                        s=0,
                                        sigma=1.,
                                        rho=0.30,
                                        randomizer_scale=1.))

        print("iteration completed ", i)

    plt.clf()
    ecdf_pivot = ECDF(np.asarray(_pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_pivot(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

if __name__ == "__main__":
    main(nsim=100)