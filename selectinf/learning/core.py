"""
Core module for learning selection

"""

from copy import copy
import functools

import numpy as np
import pandas as pd
from scipy.stats import norm as ndist
from scipy.stats import binom

from ..distributions.discrete_family import discrete_family

# local imports

from .fitters import (gbm_fit_sk,
                      random_forest_fit_sk)
from .samplers import (normal_sampler,
                       split_sampler)
from .learners import mixture_learner
try:
    from .keras_fit import keras_fit
except ImportError:
    pass

def infer_general_target(observed_outcome,
                         observed_target,
                         target_cov,
                         learner,
                         fit_probability=gbm_fit_sk,
                         fit_args={'n_estimators':500},
                         hypothesis=0,
                         alpha=0.1,
                         success_params=(1, 1),
                         B=500,
                         learning_npz=None):
    '''

    Compute a p-value (or pivot) for a target having observed `outcome` of from `algorithm` run on the original data.

    Parameters
    ----------

    observed_outcome : object
        The purported value observed, i.e. run with the original seed.

    target_cov : np.float((1, 1)) # 1 for 1-dimensional targets for now
        Covariance of target estimator

    learner : 
        Object that generates perturbed data.

    observed_target : np.float   # 1-dimensional targets for now
        Observed value of target estimator.

    fit_probability : callable
        Function to learn a probability model P(Y=1|T) based on [T, Y].

    hypothesis : np.float   # 1-dimensional targets for now
        Hypothesized value of target.

    alpha : np.float
        Level for 1 - confidence.

    B : int
        How many queries?
    '''
                              
    weight_fns, learning_data = learner.learn(fit_probability,
                                              fit_args=fit_args,
                                              check_selection=None,
                                              B=B)
    weight_fn = weight_fns[0]

    if learning_npz is not None:
        T, Y = learning_data
        npz_results = {'T':T, 'Y':Y}
        npz_results['nuisance'] = []
        npz_results['direction'] = []

    results = []
    for i in range(observed_target.shape[0]):
        cur_nuisance = observed_target - target_cov[i] / target_cov[i, i] * observed_target[i]
        cur_nuisance.shape = observed_target.shape
        direction = target_cov[i] / target_cov[i, i]

        if learning_npz is not None:
            npz_results['nuisance'].append(cur_nuisance)
            npz_results['direction'].append(direction)

        def new_weight_fn(cur_nuisance, direction, weight_fn, target_val):
            return weight_fn(np.multiply.outer(target_val, direction) + cur_nuisance[None, :])

        new_weight_fn = functools.partial(new_weight_fn, cur_nuisance, direction, weight_fn)

        results.append(_inference(observed_target[i],
                                  target_cov[i, i].reshape((1, 1)),
                                  new_weight_fn,
                                  hypothesis=hypothesis[i],
                                  alpha=alpha,
                                  success_params=success_params)[:4])

    if learning_npz is not None:
        np.savez(learning_npz, **npz_results)

    return results

def infer_set_target(observed_set,
                     features,
                     observed_target,
                     target_cov,
                     learner,
                     fit_probability=gbm_fit_sk,
                     fit_args={'n_estimators':500},
                     hypothesis=[0],
                     alpha=0.1,
                     success_params=(1, 1),
                     B=500,
                     learning_npz=None,
                     single=False):

    '''

    Compute a p-value (or pivot) for a target having observed `outcome` of `algorithm` on original data.

    Parameters
    ----------

    observed_set : set(int)
        The purported value observed when run with the original seed.

    features : [int]
        List of the elements of observed_set.

    observed_target : np.ndarray
        Statistic inference is based on.

    target_cov : np.ndarray
        (Pre-selection) covariance matrix of `observed_target`.

    learner : 
        Object that generates perturbed data.

    fit_probability : callable
        Function to learn a probability model P(Y=1|T) based on [T, Y].

    hypothesis : np.float   # 1-dimensional targets for now
        Hypothesized value of target.

    alpha : np.float
        Level for 1 - confidence.

    B : int
        How many queries?

    Notes
    -----

    This function makes the assumption that covariance in observed sampler is the 
    true covariance of S and we are looking for inference about coordinates of the mean of 

    np.linalg.inv(covariance).dot(S)

    this allows us to compute the required observed_target, cross_cov and target_cov.

    '''

    features = np.asarray(features)
    if features.shape == ():
        features = np.array([features])

    observed_set = set(observed_set)
    if np.any([f not in observed_set for f in features]):
        raise ValueError('for a set target, we can only do inference for features observed in the outcome')

    weight_fns, learning_data = learner.learn(fit_probability,
                                              fit_args=fit_args,
                                              check_selection=lambda result: np.array([f in set(result) for f in features]),
                                              B=B)

    if len(features) == 1 and success_params == (1, 1) and single:
        single_inference = _single_parameter_inference(observed_target,
                                                       target_cov,
                                                       learning_data,
                                                       learner.proposal_density, 
                                                       hypothesis=hypothesis[0],
                                                       alpha=alpha)
        return [single_inference + (None,)]
    else:

        if learning_npz is not None:
            T, Y = learning_data
            npz_results = {'T':T, 'Y':Y}
            npz_results['nuisance'] = []
            npz_results['direction'] = []

        results = []
        for i in range(observed_target.shape[0]):
            cur_nuisance = observed_target - target_cov[i] / target_cov[i, i] * observed_target[i]
            cur_nuisance.shape = observed_target.shape
            direction = target_cov[i] / target_cov[i, i]

            if learning_npz is not None:
                npz_results['nuisance'].append(cur_nuisance)
                npz_results['direction'].append(direction)

            def new_weight_fn(cur_nuisance, direction, weight_fn, target_val):
                return weight_fn(np.multiply.outer(target_val, direction) + cur_nuisance[None, :])

            new_weight_fn = functools.partial(new_weight_fn, cur_nuisance, direction, weight_fns[i])
            results.append(_inference(observed_target[i],
                                      target_cov[i, i].reshape((1, 1)),
                                      new_weight_fn,
                                      hypothesis=hypothesis[i],
                                      alpha=alpha,
                                      success_params=success_params)[:4])

        if learning_npz is not None:
            np.savez(learning_npz, **npz_results)

        return results

def infer_full_target(algorithm,
                      observed_set,
                      features,
                      observed_sampler,
                      dispersion, # sigma^2
                      fit_probability=gbm_fit_sk,
                      fit_args={'n_estimators':500},
                      hypothesis=[0],
                      alpha=0.1,
                      success_params=(1, 1),
                      B=500,
                      learner_klass=mixture_learner,
                      learning_npz=None,
                      single=False):

    '''

    Compute a p-value (or pivot) for a target having observed `outcome` of `algorithm(observed_sampler)`.

    Parameters
    ----------

    algorithm : callable
        Selection algorithm that takes a noise source as its only argument.

    observed_set : set(int)
        The purported value `algorithm(observed_sampler)`, i.e. run with the original seed.

    features : [int]
        List of the elements of observed_set.

    observed_sampler : `normal_source`
        Representation of the data used in the selection procedure.

    dispersion : float
        Scalar dispersion of the covariance of `observed_sampler`. In 
        OLS problems this is $\sigma^2$.

    fit_probability : callable
        Function to learn a probability model P(Y=1|T) based on [T, Y].

    hypothesis : np.float   # 1-dimensional targets for now
        Hypothesized value of target.

    alpha : np.float
        Level for 1 - confidence.

    B : int
        How many queries?

    Notes
    -----

    This function makes the assumption that covariance in observed sampler is the 
    true covariance of S and we are looking for inference about coordinates of the mean of 

    np.linalg.inv(covariance).dot(S)

    this allows us to compute the required observed_target, cross_cov and target_cov.

    '''

    info_inv = np.linalg.inv(observed_sampler.covariance / dispersion) # scale free, i.e. X.T.dot(X) without sigma^2
    target_cov = (info_inv[features][:, features] * dispersion)
    observed_target = info_inv[features].dot(observed_sampler.center)
    cross_cov = observed_sampler.covariance.dot(info_inv[:, features])

    learner = learner_klass(algorithm, 
                            observed_set,
                            observed_sampler, 
                            observed_target,
                            target_cov,
                            cross_cov)

    return infer_set_target(observed_set,
                            features,
                            observed_target,
                            target_cov,
                            learner,
                            fit_probability=fit_probability,
                            fit_args=fit_args,
                            hypothesis=hypothesis,
                            alpha=alpha,
                            success_params=success_params,
                            B=B,
                            learning_npz=learning_npz,
                            single=single)

def _inference(observed_target,
               target_cov,
               weight_fn, # our fitted function
               success_params=(1, 1),
               hypothesis=0,
               alpha=0.1):

    '''

    Produce p-values (or pivots) and confidence intervals having estimated a weighting function.

    The basic object here is a 1-dimensional exponential family with reference density proportional
    to 

    lambda t: scipy.stats.norm.pdf(t / np.sqrt(target_cov)) * weight_fn(t)

    Parameters
    ----------

    observed_target : float

    target_cov : np.float((1, 1))

    hypothesis : float
        Hypothesised true mean of target.

    alpha : np.float
        Level for 1 - confidence.

    Returns
    -------

    pivot : float
        Probability integral transform of the observed_target at mean parameter "hypothesis"

    confidence_interval : (float, float)
        (1 - alpha) * 100% confidence interval.

    '''

    k, m = success_params # need at least k of m successes

    target_var = target_cov[0, 0]
    target_sd = np.sqrt(target_var)
              
    target_val = (np.linspace(-20 * target_sd, 20 * target_sd, 5001) + 
                  observed_target)

    if (k, m) != (1, 1):
        weight_val = np.array([binom(m, p).sf(k-1) for p in 
                               weight_fn(target_val)])
    else:
        weight_val = np.squeeze(weight_fn(target_val))

    if DEBUG:
        import matplotlib.pyplot as plt, uuid
        plt.plot(target_val, weight_val)
        id_ = 'inference_' + str(uuid.uuid1())
        plt.savefig(id_+'_prob.png')
        plt.clf()

    weight_val *= ndist.pdf((target_val - observed_target) / target_sd)

    plt.plot(target_val, weight_val)
    plt.plot(target_val, ndist.pdf((target_val - observed_target) / target_sd), label='gaussian')
    plt.plot([hypothesis], [0], '+', color='orange')
    plt.legend()
    plt.savefig(id_+'_dens.png')
    plt.clf()

    exp_family = discrete_family(target_val, weight_val)

    pivot = exp_family.cdf((hypothesis - observed_target) 
                           / target_var, x=observed_target)
    pivot = 2 * min(pivot, 1-pivot)

    pvalue = exp_family.cdf(- observed_target / target_cov[0, 0], 
                              x=observed_target)
    pvalue = 2 * min(pvalue, 1-pvalue)

    interval = exp_family.equal_tailed_interval(observed_target, alpha=alpha)
    rescaled_interval = (interval[0] * target_var + observed_target,
                         interval[1] * target_var + observed_target)

    return pivot, rescaled_interval, pvalue, weight_fn, exp_family  # TODO: should do MLE as well does discrete_family do this?

def _single_parameter_inference(observed_target,
                                target_cov,
                                learning_data,
                                proposal_density, 
                                hypothesis=0,
                                alpha=0.1):

    '''


    lambda t: scipy.stats.norm.pdf(t / np.sqrt(target_cov)) * weight_fn(t)

    Parameters
    ----------

    observed_target : float

    target_cov : np.float((1, 1))

    hypothesis : float
        Hypothesised true mean of target.

    alpha : np.float
        Level for 1 - confidence.

    Returns
    -------

    pivot : float
        Probability integral transform of the observed_target at mean parameter "hypothesis"

    confidence_interval : (float, float)
        (1 - alpha) * 100% confidence interval.

    '''

    T, Y = learning_data
    target_val = T[Y == 1]

    target_var = target_cov[0, 0]
    target_sd = np.sqrt(target_var)
    weight_val = ndist.pdf((target_val - observed_target) / target_sd) / proposal_density(target_val.reshape((-1,1)))
    exp_family = discrete_family(target_val, weight_val)

    pivot = exp_family.cdf((hypothesis - observed_target) / target_var, x=observed_target)
    pivot = 2 * min(pivot, 1-pivot)

    pvalue = exp_family.cdf(-observed_target / target_var, x=observed_target)
    pvalue = 2 * min(pvalue, 1-pvalue)

    interval = exp_family.equal_tailed_interval(observed_target, alpha=alpha)
    rescaled_interval = (interval[0] * target_var + observed_target[0], 
                         interval[1] * target_var + observed_target[0])

    return pivot, rescaled_interval, pvalue, exp_family  # TODO: should do MLE as well does discrete_family do this?

def repeat_selection(base_algorithm, sampler, min_success, num_tries):
    """
    Repeat a set-returning selection algorithm `num_tries` times,
    returning all elements that appear at least `min_success` times.
    """

    results = {}

    for _ in range(num_tries):
        current = base_algorithm(sampler)
        for item in current:
            results.setdefault(item, 0)
            results[item] += 1
            
    final_value = []
    for key in results:
        if results[key] >= min_success:
            final_value.append(key)

    return set(final_value)


def cross_inference(learning_data, nuisance, direction, fit_probability, nref=200, fit_args={}):

    T, Y = learning_data

    idx = np.arange(T.shape[0])
    np.random.shuffle(idx)

    Tshuf, Yshuf = T[idx], Y[idx]
    reference_T = Tshuf[:nref]
    reference_Y = Yshuf[:nref]
    nrem = T.shape[0] - nref
    learning_T = Tshuf[nref:(nref+int(nrem/2))]
    learning_Y = Tshuf[nref:(nref+int(nrem/2))]
    dens_T = Tshuf[(nref+int(nrem/2)):]

    pvalues = []

    weight_fns = fit_probability(learning_T, learning_Y, **fit_args)

    for (weight_fn, 
         cur_nuisance, 
         cur_direction, 
         learn_T, 
         ref_T, 
         ref_Y,
         d_T) in zip(weight_fns, 
                     nuisance, 
                     direction, 
                     learning_T.T, 
                     reference_T.T,
                     reference_Y.T,
                     dens_T.T):
        
        def new_weight_fn(nuisance, direction, weight_fn, target_val):
                return weight_fn(np.multiply.outer(target_val, direction) + nuisance[None, :])

        new_weight_fn = functools.partial(new_weight_fn, cur_nuisance, cur_direction, weight_fn)

        weight_val = new_weight_fn(d_T)
        exp_family = discrete_family(d_T, weight_val)
        print(ref_Y)
        pval = [exp_family.cdf(0, x=t) for t, y in zip(ref_T, ref_Y) if y == 1]
        pvalues.append(pval)

    return pvalues
