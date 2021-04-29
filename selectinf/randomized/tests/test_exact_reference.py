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

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    X, Y, beta = inst(n=n,
                      p=p,
                      signal=0,
                      s=s,
                      equicorrelated=True,
                      rho=rho,
                      sigma=sigma,
                      random_signs=False)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)
    #dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    dispersion = sigma_ ** 2

    #W = 1 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * np.sqrt(dispersion)
    eps = np.random.standard_normal((n, 2000)) * Y.std()
    lam_theory = 0.7 * np.median(np.abs(X.T.dot(eps)).max(1))

    conv = const(X,
                 Y,
                 lam_theory * np.ones(p),
                 randomizer_scale=randomizer_scale * dispersion)

    signs = conv.fit()
    nonzero = signs != 0
    print("size of selected set ", nonzero.sum())

    if nonzero.sum()>0:
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=None)

        exact_grid_inf = exact_grid_inference(conv,
                                              observed_target,
                                              cov_target,
                                              cov_target_score)

        pivot = exact_grid_inf._pivots(beta_target)

        return pivot

def test_approx_ci(n=500,
                   p=100,
                   signal_fac=1.,
                   s=5,
                   sigma=2.,
                   rho=0.4,
                   randomizer_scale=1.,
                   level=0.9):

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

    W = 1 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * np.sqrt(dispersion)

    conv = const(X,
                 Y,
                 W,
                 randomizer_scale=randomizer_scale * dispersion)

    signs = conv.fit()
    nonzero = signs != 0

    if nonzero.sum()>0:

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        result, inverse_info = conv.selective_MLE(observed_target,
                                                  cov_target,
                                                  cov_target_score)[:2]

        exact_grid_inf = exact_grid_inference(conv,
                                              observed_target,
                                              cov_target,
                                              cov_target_score)

        lci, uci = exact_grid_inf._intervals(level)

    beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
    coverage = (lci < beta_target) * (uci > beta_target)
    length = uci - lci

    return np.mean(coverage), np.mean(length), np.mean(length-(3.3 * np.sqrt(np.diag(inverse_info))))

def main(nsim=300, CI=False):

    import matplotlib as mpl
    mpl.use('tkagg')
    import matplotlib.pyplot as plt
    from statsmodels.distributions.empirical_distribution import ECDF

    if CI is False:
        _pivot = []
        for i in range(nsim):
            _pivot.extend(test_approx_pivot(n=100,
                                            p=400,
                                            signal_fac=1.,
                                            s=0,
                                            sigma=1.,
                                            rho=0.30,
                                            randomizer_scale=0.7))

            print("iteration completed ", i)

        plt.clf()
        ecdf_pivot = ECDF(np.asarray(_pivot))
        grid = np.linspace(0, 1, 101)
        plt.plot(grid, ecdf_pivot(grid), c='blue', marker='^')
        plt.plot(grid, grid, 'k--')
        plt.show()

    if CI is True:
        coverage_ = 0.
        length_ = 0.
        length_diff_ = 0.
        for n in range(nsim):
            cov, len, len_diff = test_approx_ci(n=500,
                                                p=100,
                                                signal_fac=1.,
                                                s=5,
                                                sigma=3.,
                                                rho=0.50,
                                                randomizer_scale=1.)

            coverage_ += cov
            length_ += len
            length_diff_ += len_diff
            print("coverage so far ", coverage_ / (n + 1.))
            print("lengths so far ", length_ / (n + 1.), length_diff_/(n+1.))
            print("iteration completed ", n + 1)


if __name__ == "__main__":
    main(nsim=50, CI=False)