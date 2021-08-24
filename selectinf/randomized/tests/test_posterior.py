import numpy as np
import pandas as pd
from scipy.stats import norm as ndist

from ..lasso import lasso, split_lasso
from ..posterior_inference import (langevin_sampler,
                                   gibbs_sampler)

from ...base import selected_targets
from ...tests.instance import gaussian_instance, HIV_NRTI
from ...tests.flags import SET_SEED, SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsample=50, nburnin=10)
def test_Langevin(n=500,
                  p=100,
                  signal_fac=1.,
                  s=5,
                  sigma=3.,
                  rho=0.4,
                  randomizer_scale=1.,
                  nsample=1500,
                  nburnin=100):

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
                 ridge_term=0.,
                 randomizer_scale=randomizer_scale * dispersion)

    signs = conv.fit()
    nonzero = signs != 0

    beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

    target_spec = selected_targets(conv.loglike,
                                   conv.observed_soln,
                                   dispersion=dispersion)

    posterior_inf = conv.posterior(target_spec)

    samples = langevin_sampler(posterior_inf,
                               nsample=nsample,
                               nburnin=nburnin)

    lci = np.percentile(samples, 5, axis=0)
    uci = np.percentile(samples, 95, axis=0)
    coverage = (lci < beta_target) * (uci > beta_target)
    length = uci - lci

    return np.mean(coverage), np.mean(length)


@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsample=50, nburnin=10, nsim=2)
def test_coverage(nsim=100,
                  nsample=1500,
                  nburnin=100):

    cov, len = 0., 0.

    for i in range(nsim):
        cov_, len_ = test_Langevin(n=500,
                                   p=100,
                                   signal_fac=0.5,
                                   s=5,
                                   sigma=2.,
                                   rho=0.2,
                                   randomizer_scale=1.,
                                   nsample=nsample,
                                   nburnin=nburnin)

        cov += cov_
        len += len_

        print("coverage and lengths ", i, cov / (i + 1.), len / (i + 1.))


@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsample=50, nburnin=10)
def test_instance(nsample=100, nburnin=50):
    np.random.seed(10)
    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    # beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())

    target_spec = selected_targets(L.loglike,
                                   L.observed_soln,
                                   features=M,
                                   dispersion=dispersion)
    print(target_spec.dispersion, dispersion)
    
    posterior_inf = L.posterior(target_spec)

    samples = langevin_sampler(posterior_inf,
                               nsample=nsample,
                               nburnin=nburnin)

    gibbs_samples = gibbs_sampler(posterior_inf,
                                  nsample=nsample,
                                  nburnin=nburnin)[0]

    lci = np.percentile(samples, 5, axis=0)
    uci = np.percentile(samples, 95, axis=0)

    beta_target = np.linalg.pinv(X[:, M]).dot(X.dot(beta))
    coverage = (lci < beta_target) * (uci > beta_target)
    length = uci - lci

    return np.mean(coverage), np.mean(length)


@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsample=50, nburnin=10)
def test_flexible_prior1(nsample=100,
                         nburnin=50,
                         seed=0):

    np.random.seed(seed)
    
    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    # beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())

    target_spec = selected_targets(L.loglike,
                                   L.observed_soln,
                                   features=M,
                                   dispersion=dispersion)

    # default prior

    Di = 1. / (200 * np.diag(target_spec.cov_target))

    def prior(target_parameter):
        grad_prior = -target_parameter * Di
        log_prior = -0.5 * np.sum(target_parameter ** 2 * Di)
        return log_prior, grad_prior

    seed_state = np.random.get_state()
    np.random.set_state(seed_state)
    Z1 = np.random.standard_normal()

    posterior_inf1 = L.posterior(target_spec,
                                 prior=prior)

    W1 = np.random.standard_normal()
    samples1 = langevin_sampler(posterior_inf1,
                                nsample=nsample,
                                nburnin=nburnin)

    np.random.set_state(seed_state)
    Z2 = np.random.standard_normal()
    posterior_inf2 = L.posterior(target_spec)

    W2 = np.random.standard_normal()
    samples2 = langevin_sampler(posterior_inf2,
                                nsample=nsample,
                                nburnin=nburnin)
    # these two assertions essentially just check the random state
    # was run identically for samples1 and samples2 
    np.testing.assert_equal(Z1, Z2)
    np.testing.assert_equal(W1, W2)
    np.testing.assert_allclose(samples1, samples2, rtol=1.e-3)


@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsample=50, nburnin=10)
def test_flexible_prior2(nsample=1000, nburnin=50):
    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    # beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())

    target_spec = selected_targets(L.loglike,
                                   L.observed_soln,
                                   features=M,
                                   dispersion=dispersion)

    prior_var = 0.05 ** 2

    def prior(target_parameter):
        grad_prior = -target_parameter / prior_var
        log_prior = -np.linalg.norm(target_parameter) ** 2 / (2. * prior_var)
        return log_prior, grad_prior

    posterior_inf = L.posterior(target_spec,
                                prior=prior)

    adaptive_proposal = np.linalg.inv(np.linalg.inv(posterior_inf.inverse_info) +
                                      np.identity(posterior_inf.inverse_info.shape[0]) / 0.05 ** 2)
    samples = langevin_sampler(posterior_inf,
                               nsample=nsample,
                               proposal_scale=adaptive_proposal,
                               nburnin=nburnin)
    return samples



