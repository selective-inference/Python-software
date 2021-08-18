import numpy as np

from ...tests.instance import gaussian_instance
from ..lasso import lasso, selected_targets
from ..exact_reference import exact_grid_inference

def test_inf(n=500,
             p=100,
             signal_fac=1.,
             s=5,
             sigma=2.,
             rho=0.4,
             randomizer_scale=1.,
             equicorrelated=False,
             CI=True):

    while True:

        inst, const = gaussian_instance, lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=equicorrelated,
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
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs = conv.fit()
        nonzero = signs != 0
        print("size of selected set ", nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            (observed_target,
             cov_target,
             regress_target_score,
             dispersion,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

            exact_grid_inf = exact_grid_inference(conv,
                                                  observed_target,
                                                  cov_target,
                                                  regress_target_score)

            if CI is False:
                pivot = exact_grid_inf._pivots(beta_target)
                return pivot

            else:
                lci, uci = exact_grid_inf._intervals(level=0.90)
                coverage = (lci < beta_target) * (uci > beta_target)
                length = uci - lci
                mle_length = 1.65*2 * np.sqrt(np.diag(exact_grid_inf.inverse_info))
                return np.mean(coverage), np.mean(length), np.mean(mle_length)


