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
from ...tests.flags import SMALL_SAMPLES, SET_SEED
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

from ...tests.decorators import set_sampling_params_iftrue
from ..randomization import randomization
from ..reconstruction import reconstruct_full_from_internal


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
    arg = (randomization._cdf(lower) + 
           np.multiply(x, randomization._cdf(upper) - 
                       randomization._cdf(lower)))
    return randomization._ppf(arg)

def sampling_truncated_dist(lower, upper, randomization, nsamples=1000):
    uniform_samples = np.random.uniform(0,1, size=(nsamples,randomization.shape[0]))
    return inverse_truncated_cdf(uniform_samples, lower, upper, randomization)

def sample_opt_vars(X, y, active, signs, lam, epsilon, randomization, nsamples =10000):

    Xdiag = np.diag(X.T.dot(X))
    p = X.shape[1]

    unpenalized = (lam == 0) * active
    nunpenalized = unpenalized.sum()
    lower = -np.ones(p) * np.inf
    upper = -lower
    active_set = np.where(active * (lam > 0))[0]
    unpen_set = np.where(active * (lam == 0))[0]
    inactive_set = np.where(~active)[0]

    nactive = active.sum() - unpenalized.sum()
    nunpen = unpenalized.sum()
    for i in range(nactive):
        var = active_set[i]
        if lam[var] != 0:
            if signs[var]>0:
                    lower[i] = (-X[:, var].T.dot(y) + lam[var] * signs[var])
            else:
                upper[i] = (-X[:,var].T.dot(y) + lam[var] * signs[var]) 

    lower[range(nactive + nunpen, p)] = -lam[inactive_set] - X[:, inactive_set].T.dot(y)
    upper[range(nactive + nunpen, p)] = lam[inactive_set] - X[:, inactive_set].T.dot(y)

    #print(lower, 'lower')
    #print(upper, 'upper')
    omega_samples = sampling_truncated_dist(lower, 
                                            upper, 
                                            randomization, 
                                            nsamples=nsamples)

    abs_beta_samples = np.true_divide( 
                          omega_samples[:, :nactive] + 
                          X[:, active_set].T.dot(y) - 
                          lam[active_set] * signs[active_set], 
                          (epsilon + Xdiag[active_set]) * signs[active_set])
    unpen_beta_samples = np.true_divide( 
                          omega_samples[:, nactive:(nactive + nunpen)] + 
                          X[:, unpen_set].T.dot(y), 
                          (epsilon + Xdiag[unpen_set]))
    u_samples = omega_samples[:, (nactive + nunpen):] + X[:, inactive_set].T.dot(y)

    # this ordering should be correct?

    reordered_omega = np.zeros_like(omega_samples)
    reordered_omega[:, active_set] = omega_samples[:, :nactive]
    reordered_omega[:, unpen_set] = omega_samples[:, nactive:(nactive + nunpen)]
    reordered_omega[:, inactive_set] = omega_samples[:, (nactive + nunpen):]

    return np.concatenate((abs_beta_samples, unpen_beta_samples, u_samples), axis=1), reordered_omega


def orthogonal_design(n, p, s, signal, sigma, random_signs=True):
    scale = np.linspace(2, 3, p)
    X = np.identity(n)[:,:p]
    X *= scale[None, :]

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

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma


@set_seed_iftrue(SET_SEED, 200)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_conditional_law(ndraw=20000, burnin=2000, ridge_term=0.5, stepsize=None, unpenalized=False):
    """
    Checks the conditional law of opt variables given the data
    """

    results = []
    for const_info, rand in product(zip([gaussian_instance], 
                                        [lasso.gaussian]), 
                                    ['laplace', 'gaussian']):

        inst, const = const_info

        X, Y, beta = orthogonal_design(n=100, 
                                       p=9, 
                                       s=3,
                                       signal=(1,2), 
                                       sigma=1.2)[:3]
        n, p = X.shape

        W = np.linspace(2, 3, X.shape[1])
        if unpenalized:
            W[4] = 0
        else:
            W[4] = 1.e-5
        randomizer_scale = 1.
        conv = const(X, 
                     Y, 
                     W, 
                     randomizer=rand, 
                     randomizer_scale=randomizer_scale,
                     ridge_term=ridge_term,
                     parametric_cov_estimator=True)

        print(rand)
        if rand == "laplace":
            randomizer = randomization_ppf.laplace((p,), \
                             scale=randomizer_scale)
        elif rand=="gaussian":
            randomizer = randomization_ppf.isotropic_gaussian((p,), \
                             scale=randomizer_scale)

        signs = conv.fit()
        print("signs", signs)
        conv.decompose_subgradient(marginalizing_groups=np.ones(p,np.bool),
                                   conditioning_groups=np.zeros(p,np.bool))

        selected_features = conv._view.selection_variable['variables']
        q = conv._view

        opt_sampler = q.sampler # optimization_sampler(q.observed_opt_state,
#                                            q.observed_internal_state,
#                                            q.score_transform,
#                                            q.opt_transform,
#                                            q.projection,
#                                            q.grad_log_density,
#                                            q.log_density)

        S = opt_sampler.sample(ndraw,
                               burnin,
                               stepsize=stepsize)
        print(S.shape)
        print([np.mean(S[:,i]) for i in range(S.shape[1])])
        print(selected_features, 'selected')

        # let's also reconstruct the omegas to compare
        if (S.shape[1]<p):
            S = np.concatenate((S, np.zeros((S.shape[0],p-S.shape[1]))), axis=1)
        S_omega = reconstruct_opt(conv._view, S)

        opt_samples = sample_opt_vars(X, 
                                      Y, 
                                      selected_features, 
                                      signs, 
                                      W, 
                                      conv.ridge_term, 
                                      randomizer, 
                                      nsamples=ndraw)

        print([np.mean(opt_samples[0][:,i]) for i in range(p)])

        results.append((rand, S, S_omega,) + opt_samples)

    return results

    
def reconstruct_opt(query, state):
    '''
    Reconstruction of randomization at current state.
    Parameters
    ----------
    state : np.float
       State of sampler made up of `(target, opt_vars)`.
       Can be array with each row a state.

    Returns
    -------
    reconstructed : np.float
       Has shape of `opt_vars` with same number of rows
       as `state`.

    '''

    state = np.atleast_2d(state)
    if state.ndim > 2:
        raise ValueError('expecting at most 2-dimensional array')

    reconstructed = reconstruct_full_from_internal(query.opt_transform,
                                                   query.score_transform,
                                                   query.observed_internal_state,
                                                   state)

    return np.squeeze(reconstructed)


