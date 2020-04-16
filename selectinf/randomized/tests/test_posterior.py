import numpy as np
from selectinf.tests.instance import gaussian_instance
from selectinf.randomized.lasso import lasso, selected_targets
from selectinf.randomized.posterior_inference import posterior_inference_lasso

def test_sampler(n=500,
                 p=100,
                 signal_fac=1.,
                 s=5,
                 sigma=3.,
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
    W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

    conv = const(X,
                 Y,
                 W,
                 randomizer_scale=randomizer_scale * sigma_)

    signs = conv.fit()
    nonzero = signs != 0

    beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
    dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(conv.loglike,
                                      conv._W,
                                      nonzero,
                                      dispersion=dispersion)

    posterior_inf = posterior_inference_lasso(observed_target,
                                              cov_target,
                                              cov_target_score,
                                              conv.observed_opt_state,
                                              conv.cond_mean,
                                              conv.cond_cov,
                                              conv.logdens_linear,
                                              conv.A_scaling,
                                              conv.b_scaling,
                                              observed_target)

    samples = posterior_inf.posterior_sampler(nsample=2000, nburnin=200, step=1.)
    lci = np.percentile(samples, 5, axis=0)
    uci = np.percentile(samples, 95, axis=0)
    coverage = (lci < beta_target) * (uci > beta_target)
    length = uci - lci

    print("check ", coverage, length)


test_sampler()
