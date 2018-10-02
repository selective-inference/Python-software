from copy import copy
import functools

import numpy as np
from scipy.stats import norm as ndist
from scipy.stats import binom

from selection.distributions.discrete_family import discrete_family

# local imports

from fitters import (logit_fit,
                     probit_fit)

# randomization mechanism

class normal_sampler(object):

    """
    Our basic model for noise, and input to 
    selection algorithms. This represents
    Gaussian data with a center, e.g. X.T.dot(y)
    in linear regression and a covariance Sigma.

    This object emits noisy versions of `center` as

    center + scale * N(0, Sigma)

    """
    def __init__(self, center, covariance):
        '''
        Parameters
        ----------

        center : np.float(p)
            Center of Gaussian noise source.

        covariance : np.float((p, p))
            Covariance of noise added (up to scale factor).

        '''
        (self.center,
         self.covariance) = (np.asarray(center),
                             np.asarray(covariance))
        self.cholT = np.linalg.cholesky(self.covariance).T
        self.shape = self.center.shape

    def __call__(self, size=None, scale=1.):

        '''

        Parameters
        ----------

        size : tuple or int
            How many copies to draw

        scale : float
            Scale (in data units) applied to unitless noise before adding.

        Returns
        -------

        noisy_sample : np.float

        Generate noisy version of the center. With scale==0.,
        return the full center.

        TODO: for some calculations, a log of each call would be helpful
        for constructing UMVU, say.

        '''

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        if self.shape == ():
            _shape = (1,)
        else:
            _shape = self.shape
        return scale * np.squeeze(np.random.standard_normal(size + _shape).dot(self.cholT)) + self.center

class split_sampler(object):

    """
    Data splitting noise source.
    This is approximately
    Gaussian with center np.sum(sample_stat, 0)
    and noise suitably scaled, depending
    on splitting fraction.

    This object emits noisy versions of `center` as

    center + scale * N(0, Sigma)

    """

    def __init__(self, sample_stat, covariance): # covariance of sum of rows
        '''
        Parameters
        ----------

        sample_stat : np.float((n, p))
             Data matrix. In linear regression this is X * y[:, None]

        covariance : np.float((p, p))
             Covariance of np.sum(sample_stat, 0). Could be computed
             e.g. by bootstrap or parametric method given a design X.
        '''
        self.sample_stat = np.asarray(sample_stat)
        self.nsample = self.sample_stat.shape[0]
        self.center = np.sum(self.sample_stat, 0)
        self.covariance = covariance
        self.shape = self.center.shape

    def __call__(self, size=None, scale=0.5):

        '''
        Parameters
        ----------

        size : tuple or int
            How many copies to draw

        scale : float
            Scale (in data units) applied to unitless noise before adding.

        Returns
        -------

        noisy_sample : np.float

        Generate noisy version of the center. With scale==0.,
        return the full center.

        The equivalent data splitting fraction is 1 / (scale**2 + 1).
        Argument is kept as `scale` instead of `frac` so that the general
        learning algorithm can replace this `splitter_source` with a corresponding
        `normal_source`.

        TODO: for some calculations, a log of each call would be helpful
        for constructing UMVU, say.
        '''

        # (1 - frac) / frac = scale**2

        frac = 1 / (scale**2 + 1)

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        if self.shape == ():
            _shape = (1,)
        else:
            _shape = self.shape

        final_sample = []
        idx = np.arange(self.nsample)
        for _ in range(np.product(size)):
            sample_ = self.sample_stat[np.random.choice(idx, int(frac * self.nsample), replace=False)]
            final_sample.append(np.sum(sample_, 0) / frac) # rescale to the scale of a sum of nsample rows
        val = np.squeeze(np.array(final_sample).reshape(size + _shape))
        return val

def infer_general_target(algorithm,
                         observed_outcome,
                         observed_sampler,
                         observed_target,
                         cross_cov,
                         target_cov,
                         fit_probability=probit_fit,
                         fit_args={'df':20},
                         hypothesis=0,
                         alpha=0.1,
                         success_params=(1, 1),
                         B=500):
    '''

    Compute a p-value (or pivot) for a target having observed `outcome` of `algorithm(observed_sampler)`.

    Parameters
    ----------

    algorithm : callable
        Selection algorithm that takes a noise source as its only argument.

    observed_outcome : object
        The purported value `algorithm(observed_sampler)`, i.e. run with the original seed.

    observed_sampler : `normal_source`
        Representation of the data used in the selection procedure.

    cross_cov : np.float((*,1)) # 1 for 1-dimensional targets for now
        Covariance between `observed_sampler.center` and target estimator.

    target_cov : np.float((1, 1)) # 1 for 1-dimensional targets for now
        Covariance of target estimator

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

    target_sd = np.sqrt(target_cov[0, 0])

    # could use an improvement here...

    def learning_proposal(sd=target_sd, center=observed_target):
        scale = np.random.choice([0.25, 0.5, 1, 1.5, 2, 3], 1)
        return np.random.standard_normal() * sd * scale + center

    weight_fn = learn_weights(algorithm, 
                              observed_outcome,
                              observed_sampler, 
                              observed_target,
                              target_cov,
                              cross_cov,
                              learning_proposal, 
                              fit_probability,
                              fit_args=fit_args,
                              B=B)

    return _inference(observed_target,
                      target_cov,
                      weight_fn,
                      hypothesis=hypothesis,
                      alpha=alpha,
                      success_params=success_params)

def infer_full_target(algorithm,
                      observed_set,
                      feature,
                      observed_sampler,
                      dispersion, # sigma^2
                      min_success=None,
                      ntries=None,
                      fit_probability=probit_fit,
                      fit_args={'df':20},
                      hypothesis=0,
                      alpha=0.1,
                      success_params=(1, 1),
                      B=500):

    '''

    Compute a p-value (or pivot) for a target having observed `outcome` of `algorithm(observed_sampler)`.

    Parameters
    ----------

    algorithm : callable
        Selection algorithm that takes a noise source as its only argument.

    observed_set : set(int)
        The purported value `algorithm(observed_sampler)`, i.e. run with the original seed.

    feature : int
        One of the elements of observed_set.

    observed_sampler : `normal_source`
        Representation of the data used in the selection procedure.

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
    target_cov = (info_inv[feature, feature] * dispersion).reshape((1, 1))
    observed_target = np.squeeze(info_inv[feature].dot(observed_sampler.center))
    cross_cov = observed_sampler.covariance.dot(info_inv[feature]).reshape((-1,1))

    observed_set = set(observed_set)
    if feature not in observed_set:
        raise ValueError('for full target, we can only do inference for features observed in the outcome')

    target_sd = np.sqrt(target_cov[0, 0])

    # could use an improvement here...

    def learning_proposal(sd=target_sd, center=observed_target):
        scale = np.random.choice([0.5, 1, 1.5, 2], 1)
        return np.random.standard_normal() * sd * scale + center

    weight_fn = learn_weights(algorithm, 
                              observed_set,
                              observed_sampler, 
                              observed_target,
                              target_cov,
                              cross_cov,
                              learning_proposal, 
                              fit_probability,
                              fit_args=fit_args,
                              check_selection=lambda result: feature in set(result),
                              B=B)

    return _inference(observed_target,
                      target_cov,
                      weight_fn,
                      min_success=min_success,
                      ntries=ntries,
                      hypothesis=hypothesis,
                      alpha=alpha,
                      success_params=success_params)


def learn_weights(algorithm, 
                  observed_outcome,
                  observed_sampler, 
                  observed_target,
                  target_cov,
                  cross_cov,
                  learning_proposal, 
                  fit_probability, 
                  fit_args={},
                  B=500,
                  check_selection=None):
    """
    Learn a function 

    P(Y=1|T, N=S-c*T)

    where N is the sufficient statistic corresponding to nuisance parameters and T is our target.
    The random variable Y is 

    Y = check_selection(algorithm(new_sampler))

    That is, we perturb the center of observed_sampler along a ray (or higher-dimensional affine
    subspace) and rerun the algorithm, checking to see if the test `check_selection` passes.

    For full model inference, `check_selection` will typically check to see if a given feature
    is still in the selected set. For general targets, we will typically condition on the exact observed value 
    of `algorithm(observed_sampler)`.

    Parameters
    ----------

    algorithm : callable
        Selection algorithm that takes a noise source as its only argument.

    observed_set : set(int)
        The purported value `algorithm(observed_sampler)`, i.e. run with the original seed.

    feature : int
        One of the elements of observed_set.

    observed_sampler : `normal_source`
        Representation of the data used in the selection procedure.

    learning_proposal : callable
        Proposed position of new T to add to evaluate algorithm at.

    fit_probability : callable
        Function to learn a probability model P(Y=1|T) based on [T, Y].

    B : int
        How many queries?

    """
    S = selection_stat = observed_sampler.center

    new_sampler = normal_sampler(observed_sampler.center.copy(),
                                 observed_sampler.covariance.copy())

    if check_selection is None:
        check_selection = lambda result: result == observed_outcome

    direction = cross_cov.dot(np.linalg.inv(target_cov).reshape((1,1))) # move along a ray through S with this direction

    learning_Y, learning_T = [], []

    def random_meta_algorithm(new_sampler, algorithm, check_selection, T):
         new_sampler.center = S + direction.dot(T - observed_target)
         new_result = algorithm(new_sampler)
         return check_selection(new_result)

    random_algorithm = functools.partial(random_meta_algorithm, new_sampler, algorithm, check_selection)

    # this is the "active learning bit"
    # START

    for _ in range(B):
         T = learning_proposal()      # a guess at informative distribution for learning what we want
         Y = random_algorithm(T)

         learning_Y.append(Y)
         learning_T.append(T)

    learning_Y = np.array(learning_Y, np.float)
    learning_T = np.squeeze(np.array(learning_T, np.float))

    conditional_law = fit_probability(learning_T, learning_Y, **fit_args)
    return conditional_law

# Private functions

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

    target_sd = np.sqrt(target_cov[0, 0])
              
    target_val = np.linspace(-20 * target_sd, 20 * target_sd, 5001) + observed_target

    if (k, m) != (1, 1):
        weight_val = np.array([binom(m, p).sf(k-1) for p in weight_fn(target_val)])
    else:
        weight_val = weight_fn(target_val)

    weight_val *= ndist.pdf(target_val / target_sd)
    exp_family = discrete_family(target_val, weight_val)

    pivot = exp_family.cdf(hypothesis / target_cov[0, 0], x=observed_target)
    pivot = 2*min(pivot, 1-pivot)

    interval = exp_family.equal_tailed_interval(observed_target, alpha=alpha)
    rescaled_interval = (interval[0] * target_cov[0, 0], interval[1] * target_cov[0, 0])

    return pivot, rescaled_interval   # TODO: should do MLE as well does discrete_family do this?

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
