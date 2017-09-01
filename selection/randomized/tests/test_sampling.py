from itertools import product
import numpy as np
import nose.tools as nt

from selection.randomized.convenience import lasso, step, threshold
from selection.randomized.query import optimization_sampler
from selection.tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from selection.tests.flags import SMALL_SAMPLES
from selection.tests.decorators import set_sampling_params_iftrue
from scipy.stats import t as tdist


def inverse_truncated_cdf(x, lower, upper, randomization):
    #if (x<0 or x>1):
    #    raise ValueError("argument for cdf inverse should be in (0,1)")
    arg = randomization._cdf(lower) + np.multiply(x, randomization._cdf(upper) - randomization._cdf(lower))
    return randomization._ppf(arg)


def sampling_truncated_dist(lower, upper, randomization, nsamples=1000):
    uniform_samples = np.random.uniform(0,1, size=(nsamples,randomization.shape[0]))
    samples = np.zeros((nsamples, randomization.shape[0]))
    for i in range(nsamples):
        samples[i,:] = inverse_truncated_cdf(uniform_samples[i,:], lower, upper, randomization)
    return samples


def sample_opt_vars(X, y, active, signs, lam, epsilon, randomization, nsamples =1000):
    p = X.shape[1]
    nactive = active.sum()
    lower = np.zeros(p)
    upper = np.zeros(p)
    active_set = np.where(active)[0]

    for i in range(nactive):
        if signs[i]>0:
            lower[i] = -np.dot(X[:, active_set[i]].T,y) + lam*signs[i]
            upper[i] = np.inf
        else:
            lower[i] = -np.inf
            upper[i] = -np.dot(X[:,active_set[i]].T,y) + lam*signs[i]

    lower[range(nactive,p)] = -lam-np.dot(X[:, ~active].T, y)
    upper[range(nactive,p)]= lam-np.dot(X[:,~active].T, y)

    omega_samples = sampling_truncated_dist(lower, upper, randomization)

    beta_samples = (omega_samples[:,:nactive]+np.dot(X[:,active].T, y))/(epsilon+1)
    u_samples = (omega_samples[:, nactive:]+np.dot(X[:,~active].T, y))/lam

    return np.concatenate((beta_samples, u_samples), axis=1)

def orthogonal_design(n, p, s, signal, sigma, df=np.inf, random_signs=False):
    X = np.identity(n)[:,:p]

    beta = np.zeros(p)
    signal = np.atleast_1d(signal)
    if signal.shape == (1,):
        beta[:s] = signal[0]
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    np.random.shuffle(beta)

    active = np.zeros(p, np.bool)
    active[beta != 0] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise(n, df)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma




@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_optimization_sampler(ndraw=1000, burnin=200):

    cls = lasso
    for const_info, rand in product(zip([gaussian_instance], [cls.gaussian]), ['laplace']):

        inst, const = const_info

        X, Y = orthogonal_design(n=100, p=10, s=0, signal=2, sigma=1)[:2]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 1
        conv = const(X, Y, W, randomizer=rand)
        signs = conv.fit()
        print("signs", signs)

        marginalizing_groups = np.zeros(p, np.bool)
        #marginalizing_groups[:int(p/2)] = True
        conditioning_groups = ~marginalizing_groups
        #conditioning_groups[-int(p/4):] = False

        selected_features = conv._view.selection_variable['variables']

        #conv.summary(selected_features,
        #             ndraw=ndraw,
        #             burnin=burnin,
        #             compute_intervals=True)

        #conv.decompose_subgradient(marginalizing_groups=marginalizing_groups,
        #                           conditioning_groups=conditioning_groups)
        conv._queries.setup_sampler(form_covariances=None)
        conv._queries.setup_opt_state()
        target_sampler = optimization_sampler(conv._queries)

        S = target_sampler.sample(ndraw,
                                  burnin,
                                  stepsize=1.e-3)
        print(S.shape)
        print([np.mean(S[:,i]) for i in range(p)])

        opt_samples = sample_opt_vars(X,Y, selected_features, signs, W[0], conv.ridge_term,
                                      conv.randomizer, nsamples =1000)

        print([np.mean(opt_samples[:,i]) for i in range(p)])



test_optimization_sampler()