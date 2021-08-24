import numpy as np

from ...tests.instance import gaussian_instance
from ..lasso import lasso
from ...base import selected_targets
from ..approx_reference import approximate_grid_inference

def test_inf(n=500,
             p=100,
             signal_fac=1.,
             s=5,
             sigma=2.,
             rho=0.4,
             randomizer_scale=1.,
             equicorrelated=False,
             useIP=True,
             CI=False):

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:

        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=equicorrelated,
                          rho=rho,
                          sigma=sigma,
                          random_signs=False)[:3]

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
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("no of variables selected ", nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            print(target_spec)
            
            approximate_grid_inf = approximate_grid_inference(conv,
                                                              target_spec,
                                                              useIP=useIP)

            if CI is False:
                pivot = approximate_grid_inf._approx_pivots(beta_target)

                return pivot
            else:
                lci, uci = approximate_grid_inf._approx_intervals(level=0.90)
                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
                coverage = (lci < beta_target) * (uci > beta_target)
                length = uci - lci

                return np.mean(coverage), np.mean(length)


