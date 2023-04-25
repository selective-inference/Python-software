import numpy as np
import pandas as pd
from scipy.stats import norm as ndist

from ..lasso import split_lasso
from ..posterior_inference import (langevin_sampler,
                                   gibbs_sampler)

from ...base import selected_targets
from ...tests.instance import HIV_NRTI
from ...tests.flags import SET_SEED, SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsample=50, nburnin=10)
def test_hiv_data(nsample=10000,
                  nburnin=500,
                  level=0.90,
                  split_proportion=0.50,
                  seedn=1):
    np.random.seed(seedn)

    alpha = (1 - level) / 2
    Z_quantile = ndist.ppf(1 - alpha)

    X, Y, _ = HIV_NRTI(standardize=True)
    Y *= 15
    n, p = X.shape
    X /= np.sqrt(n)

    ols_fit = np.linalg.pinv(X).dot(Y)
    _sigma = np.linalg.norm(Y - X.dot(ols_fit)) / np.sqrt(n - p - 1)

    const = split_lasso.gaussian

    dispersion = _sigma ** 2

    W = 1 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * _sigma

    conv = const(X,
                 Y,
                 W,
                 proportion=split_proportion)

    signs = conv.fit()
    nonzero = signs != 0

    target_spec = selected_targets(conv.loglike,
                                   conv.observed_soln,
                                   dispersion=dispersion)

    mle, inverse_info = conv.selective_MLE(target_spec,
                                           level=level,
                                           solve_args={'tol': 1.e-12})[:2]

    approx_inf = conv.approximate_grid_inference(target_spec,
                                                 useIP=True)

    posterior_inf = conv.posterior(target_spec,
                                   dispersion=dispersion)

    samples_langevin = langevin_sampler(posterior_inf,
                                        nsample=nsample,
                                        nburnin=nburnin,
                                        step=1.)

    lower_langevin = np.percentile(samples_langevin, int(alpha * 100), axis=0)
    upper_langevin = np.percentile(samples_langevin, int((1 - alpha) * 100), axis=0)

    samples_gibbs, scale_gibbs = gibbs_sampler(posterior_inf,
                                               nsample=nsample,
                                               nburnin=nburnin)

    lower_gibbs = np.percentile(samples_gibbs, int(alpha * 100), axis=0)
    upper_gibbs = np.percentile(samples_gibbs, int((1 - alpha) * 100), axis=0)

    naive_est = np.linalg.pinv(X[:, nonzero]).dot(Y)
    naive_cov = dispersion * np.linalg.inv(X[:, nonzero].T.dot(X[:, nonzero]))
    naive_intervals = np.vstack([naive_est - Z_quantile * np.sqrt(np.diag(naive_cov)),
                                 naive_est + Z_quantile * np.sqrt(np.diag(naive_cov))]).T

    X_split = X[~conv._selection_idx, :]
    Y_split = Y[~conv._selection_idx]
    split_est = np.linalg.pinv(X_split[:, nonzero]).dot(Y_split)
    split_cov = dispersion * np.linalg.inv(X_split[:, nonzero].T.dot(X_split[:, nonzero]))
    split_intervals = np.vstack([split_est - Z_quantile * np.sqrt(np.diag(split_cov)),
                                 split_est + Z_quantile * np.sqrt(np.diag(split_cov))]).T

    print("lengths: adjusted intervals Langevin, Gibbs, MLE1, MLE2, approx ",
          np.mean(upper_langevin - lower_langevin),
          np.mean(upper_gibbs - lower_gibbs),
          np.mean((2 * Z_quantile) * np.sqrt(np.diag(posterior_inf.inverse_info))),
          np.mean(mle['upper_confidence'] - mle['lower_confidence']),
          np.mean(approx_inf['upper_confidence'] - approx_inf['lower_confidence'])
          )

    print("lengths: naive intervals ", np.mean(naive_intervals[:, 1] - naive_intervals[:, 0]))

    print("lengths: split intervals ", np.mean(split_intervals[:, 1] - split_intervals[:, 0]))

    scale_interval = np.percentile(scale_gibbs, [alpha * 100, (1 - alpha) * 100])
    output = pd.DataFrame({'Langevin_lower_credible': lower_langevin,
                           'Langevin_upper_credible': upper_langevin,
                           'Gibbs_lower_credible': lower_gibbs,
                           'Gibbs_upper_credible': upper_gibbs,
                           'MLE_lower_confidence': mle['lower_confidence'],
                           'MLE_upper_confidence': mle['upper_confidence'],
                           'approx_lower_confidence': approx_inf['lower_confidence'],
                           'approx_upper_confidence': approx_inf['upper_confidence'],
                           'Split_lower_confidence': split_intervals[:, 0],
                           'Split_upper_confidence': split_intervals[:, 1],
                           'Naive_lower_confidence': naive_intervals[:, 0],
                           'Naive_upper_confidence': naive_intervals[:, 1]
                           })

    return output, scale_interval, _sigma

