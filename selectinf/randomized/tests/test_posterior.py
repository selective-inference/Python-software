import numpy as np

from ...tests.instance import gaussian_instance
from ..lasso import lasso, selected_targets
from ..posterior_inference import (posterior,
                                   langevin_sampler,
                                   gibbs_sampler)

def test_Langevin(n=500,
                  p=100,
                  signal_fac=1.,
                  s=5,
                  sigma=3.,
                  rho=0.4,
                  randomizer_scale=1.,
                  nsample=100,
                  nburnin=50):

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

    beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(conv.loglike,
                                      conv._W,
                                      nonzero,
                                      dispersion=dispersion)

    posterior_inf = conv.posterior(observed_target,
                                   cov_target,
                                   cov_target_score,
                                   dispersion=dispersion)

    samples = langevin_sampler(posterior_inf,
                               nsample=nsample,
                               nburnin=nburnin)

    gibbs_samples = gibbs_sampler(posterior_inf,
                                  nsample=nsample,
                                  nburnin=nburnin)

    lci = np.percentile(samples, 5, axis=0)
    uci = np.percentile(samples, 95, axis=0)
    coverage = (lci < beta_target) * (uci > beta_target)
    length = uci - lci

    return np.mean(coverage), np.mean(length)

def test_instance(nsample=100, nburnin=50):

    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    #beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())
    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(L.loglike,
                                      L._W,
                                      M,
                                      dispersion=dispersion)

    posterior_inf = L.posterior(observed_target,
                                cov_target,
                                cov_target_score,
                                dispersion=dispersion)

    samples = langevin_sampler(posterior_inf,
                               nsample=nsample,
                               nburnin=nburnin)

    gibbs_samples = gibbs_sampler(posterior_inf,
                                  nsample=nsample,
                                  nburnin=nburnin)

    lci = np.percentile(samples, 5, axis=0)
    uci = np.percentile(samples, 95, axis=0)

    beta_target = np.linalg.pinv(X[:, M]).dot(X.dot(beta))
    coverage = (lci < beta_target) * (uci > beta_target)
    length = uci - lci

    return np.mean(coverage), np.mean(length)


def test_flexible_prior1(nsample=100, nburnin=50):

    np.random.seed(0)
    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    #beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())
    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(L.loglike,
                                      L._W,
                                      M,
                                      dispersion=dispersion)

    def prior(target_parameter):
        grad_prior = -target_parameter / 100
        log_prior = -np.linalg.norm(target_parameter)**2 /(2. * 100)
        return grad_prior, log_prior

    seed_state = np.random.get_state()
    np.random.set_state(seed_state)
    Z1 = np.random.standard_normal()
    posterior_inf1 = L.posterior(observed_target,
                                 cov_target,
                                 cov_target_score,
                                 dispersion=dispersion,
                                 prior=prior)
    W1 = np.random.standard_normal()
    samples1 = langevin_sampler(posterior_inf1,
                                nsample=nsample,
                                nburnin=nburnin)

    np.random.set_state(seed_state)
    Z2 = np.random.standard_normal()
    posterior_inf2 = L.posterior(observed_target,
                                 cov_target,
                                 cov_target_score,
                                 dispersion=dispersion)
    W2 = np.random.standard_normal()
    samples2 = langevin_sampler(posterior_inf2,
                                nsample=nsample,
                                nburnin=nburnin)
    np.testing.assert_equal(Z1, Z2)
    np.testing.assert_equal(W1, W2)
    np.testing.assert_allclose(samples1, samples2, rtol=1.e-3)
    

def test_flexible_prior2(nsample=1000, nburnin=50):

    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    #beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())
    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(L.loglike,
                                      L._W,
                                      M,
                                      dispersion=dispersion)

    prior_var = 0.05**2
    def prior(target_parameter):
        grad_prior = -target_parameter / prior_var
        log_prior = -np.linalg.norm(target_parameter)**2 /(2. * prior_var)
        return grad_prior, log_prior

    posterior_inf = L.posterior(observed_target,
                                cov_target,
                                cov_target_score,
                                dispersion=dispersion,
                                prior=prior)
    adaptive_proposal = np.linalg.inv(np.linalg.inv(posterior_inf.inverse_info) +
                                      np.identity(posterior_inf.inverse_info.shape[0]) / 0.05**2)
    samples = langevin_sampler(posterior_inf,
                               nsample=nsample,
                               proposal_scale=adaptive_proposal,
                               nburnin=nburnin)
    return samples
    

def main(ndraw=10):

    coverage_ = 0.
    length_ = 0.
    for n in range(ndraw):
        # cov, len = test_Langevin(n=500,
        #                          p=200,
        #                          signal_fac=1.5,
        #                          s=5,
        #                          sigma=2.,
        #                          rho=0.2,
        #                          randomizer_scale=1.
        #                          )

        cov, len = test_instance(nsample=2000,
                                 nburnin=100)

        coverage_ += cov
        length_ += len

        print("coverage so far ", coverage_ / (n + 1.))
        print("lengths so far ", length_ / (n + 1.))
        print("iteration completed ", n + 1)


if __name__ == "__main__":
    main()

