import numpy as np

from ...tests.instance import gaussian_group_instance
from ..approx_reference_grouplasso import group_lasso, approximate_grid_inference

def test_approx_pivot(n=500,
                      p=200,
                      signal_fac=0.1,
                      sgroup=3,
                      groups=np.arange(50).repeat(4),
                      sigma=3.,
                      rho=0.3,
                      randomizer_scale=1,
                      weight_frac=1.5):

    while True:

        inst, const = gaussian_group_instance, group_lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          sgroup=sgroup,
                          groups=groups,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        penalty_weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

        conv = const(X,
                     Y,
                     groups,
                     penalty_weights,
                     randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs, _ = conv.fit()
        nonzero = signs != 0
        print("number of selected variables ", nonzero.sum())

        if nonzero.sum() > 0:
            conv._setup_implied_gaussian()

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            approximate_grid_inf = approximate_grid_inference(conv,
                                                              dispersion)

            pivot = approximate_grid_inf._approx_pivots(beta_target)

            return pivot


def main(nsim=300, CI = False):

    import matplotlib as mpl
    mpl.use('tkagg')
    import matplotlib.pyplot as plt
    from statsmodels.distributions.empirical_distribution import ECDF
    if CI is False:
        _pivot = []
        for i in range(nsim):
            _pivot.extend(test_approx_pivot(n=500,
                                            p=100,
                                            signal_fac=1.,
                                            sgroup=0,
                                            groups=np.arange(25).repeat(4),
                                            sigma=2.,
                                            rho=0.20,
                                            randomizer_scale=0.5,
                                            weight_frac=1.2))

            print("iteration completed ", i)

        plt.clf()
        ecdf_MLE = ECDF(np.asarray(_pivot))
        grid = np.linspace(0, 1, 101)
        plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
        plt.plot(grid, grid, 'k--')
        plt.show()

if __name__ == "__main__":

    main(nsim=50, CI = False)
