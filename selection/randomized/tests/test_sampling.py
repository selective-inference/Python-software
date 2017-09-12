from itertools import product
import nose.tools as nt

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import laplace, logistic, norm as ndist

from ..convenience import lasso, step, threshold
from ..query import optimization_sampler
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from ...tests.flags import SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

from ...tests.decorators import set_sampling_params_iftrue
from ..randomization import randomization


class randomization_ppf(randomization):

    def __init__(self, rand, ppf):

        self._cdf = rand._cdf
        self._ppf = ppf
        self.shape = rand.shape

    @staticmethod
    def laplace(shape, scale):
        ppf = lambda x: laplace.ppf(x, loc=0, scale=scale)
        rand = randomization.laplace(shape, scale)
        return randomization_ppf(rand, ppf)

    @staticmethod
    def isotropic_gaussian(shape, scale):
        ppf = lambda x: ndist.ppf(x, loc=0., scale=scale)
        rand = randomization.isotropic_gaussian(shape, scale)
        return randomization_ppf(rand, ppf)


def inverse_truncated_cdf(x, lower, upper, randomization):
    #if (x<0 or x>1):
    #    raise ValueError("argument for cdf inverse should be in (0,1)")
    arg = randomization._cdf(lower) + np.multiply(x, randomization._cdf(upper) - randomization._cdf(lower))
    return randomization._ppf(arg)
    #return randomization._ppf(arg)

def sampling_truncated_dist(lower, upper, randomization, nsamples=1000):
    uniform_samples = np.random.uniform(0,1, size=(nsamples,randomization.shape[0]))
    samples = np.zeros((nsamples, randomization.shape[0]))
    for i in range(nsamples):
        samples[i,:] = inverse_truncated_cdf(uniform_samples[i,:], lower, upper, randomization)
    return samples

def sample_opt_vars(X, y, active, signs, lam, epsilon, randomization, nsamples =10000):
    p = X.shape[1]
    nactive = active.sum()
    lower = np.zeros(p)
    upper = np.zeros(p)
    active_set = np.where(active)[0]

    for i in range(nactive):
        var = active_set[i]
        if signs[var]>0:
            lower[i] = -np.dot(X[:, var].T,y) + lam*signs[var]
            upper[i] = np.inf
        else:
            lower[i] = -np.inf
            upper[i] = -np.dot(X[:,var].T,y) + lam*signs[var]

    lower[range(nactive,p)] = -lam-np.dot(X[:, ~active].T, y)
    upper[range(nactive,p)]= lam-np.dot(X[:,~active].T, y)

    omega_samples = sampling_truncated_dist(lower, upper, randomization, nsamples=nsamples)

    abs_beta_samples = np.true_divide(omega_samples[:,:nactive]+np.dot(X[:,active].T, y)-lam*signs[active], (epsilon+1)*signs[active])
    u_samples = (omega_samples[:, nactive:]+np.dot(X[:,~active].T, y))

    return np.concatenate((abs_beta_samples, u_samples), axis=1)


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



@set_seed_iftrue(True, 200)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_sampling(ndraw=20000, burnin=2000):

    cls = lasso
    for const_info, rand in product(zip([gaussian_instance], [cls.gaussian]), ['laplace', 'gaussian']):

        inst, const = const_info

        X, Y = orthogonal_design(n=100, p=10, s=0, signal=2, sigma=1)[:2]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 1
        randomizer_scale =1.
        conv = const(X, Y, W, randomizer=rand, randomizer_scale = randomizer_scale)

        print(rand)
        if rand == "laplace":
            randomizer = randomization_ppf.laplace((p,), scale=randomizer_scale)
        elif rand=="gaussian":
            randomizer = randomization_ppf.isotropic_gaussian((p,),scale=randomizer_scale)

        signs = conv.fit()
        print("signs", signs)

        selected_features = conv._view.selection_variable['variables']

        conv._queries.setup_sampler(form_covariances=None)
        conv._queries.setup_opt_state()
        target_sampler = optimization_sampler(conv._queries)

        S = target_sampler.sample(ndraw,
                                  burnin,
                                  stepsize=None)
        print(S.shape)
        print([np.mean(S[:,i]) for i in range(p)])

        opt_samples = sample_opt_vars(X,Y, selected_features, signs, W[0], conv.ridge_term, randomizer, nsamples =1000)

        print([np.mean(opt_samples[:,i]) for i in range(p)])

    return None