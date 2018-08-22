import numpy as np
from copy import copy
from fitters import (logit_fit,
                     probit_fit)
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist

# randomization mechanism

class normal_sampler(object):

    def __init__(self, center, covariance):
        (self.center,
         self.covariance) = (np.asarray(center),
                             np.asarray(covariance))
        self.cholT = np.linalg.cholesky(self.covariance).T
        self.shape = self.center.shape

    def __call__(self, scale=1., size=None):

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        if self.shape == ():
            _shape = (1,)
        else:
            _shape = self.shape
        return scale * np.squeeze(np.random.standard_normal(size + _shape).dot(self.cholT)) + self.center

    def __copy__(self):
        return normal_sampler(self.center.copy(),
                              self.covariance.copy())

def learn_weights(algorithm, 
                  observed_outcome,
                  observed_sampler, 
                  observed_target,
                  target_cov,
                  cross_cov,
                  learning_proposal, 
                  fit_probability, 
                  B=15000,
                  set_mode=False):
    """
    The algorithm to learn P(Y=1|T)
    """
    S = selection_stat = observed_sampler.center
    new_sampler = copy(observed_sampler)

    if set_mode:
        observed_set = set(observed_outcome)

    direction = cross_cov.dot(np.linalg.inv(target_cov)) # move along a ray through S with this direction

    learning_sample = []
    for _ in range(B):
         T = learning_proposal()      # a guess at informative distribution for learning what we want
         new_sampler = copy(observed_sampler)
         new_sampler.center = S + direction.dot(T - observed_target)
         new_result = algorithm(new_sampler)
         if not set_mode:
             Y = new_result == observed_outcome
         else:
             Y = new_result in observed_set
         learning_sample.append((T, Y))
    learning_sample = np.array(learning_sample)

    T, Y = learning_sample.T
    conditional_law = fit_probability(T, Y)
    return conditional_law

def infer_general_target(algorithm,
                         observed_outcome,
                         observed_sampler,
                         observed_target,
                         cross_cov,
                         target_cov,
                         fitter=logit_fit,
                         hypothesis=0,
                         alpha=0.1):

    target_sd = np.sqrt(target_cov[0,0])

    def learning_proposal(sd=target_sd, center=0):
        scale = np.random.choice([0.5, 1, 1.5, 2], 1)[0]
        return np.random.standard_normal() * sd * scale + center

    weight_fn = learn_weights(algorithm, 
                              observed_outcome,
                              observed_sampler, 
                              observed_target,
                              target_cov,
                              cross_cov,
                              learning_proposal, 
                              fitter,
                              set_mode=False)

    # let's form the pivot

    target_val = np.linspace(-20 * target_sd, 20 * target_sd, 5001) + observed_target
    weight_val = weight_fn(target_val) 
    exp_family = discrete_family(target_val, weight_val)  
    print(hypothesis, observed_target)
    pivot = exp_family.cdf(hypothesis / target_cov[0, 0], x=observed_target)
    interval = exp_family.equal_tailed_interval(observed_target, alpha=alpha)
    rescaled_interval = (interval[0] * target_cov[0, 0], interval[1] * target_cov[0, 0])

    return pivot, rescaled_interval   # TODO: should do MLE as well does discrete_family do this?

def infer_full_target(algorithm,
                      observed_set,
                      feature,
                      observed_sampler,
                      fitter=logit_fit,
                      hypothesis=0,
                      alpha=0.1):

    # this makes assumption that covariance in observed sampler is the 
    # true covariance of S
    # and we are looking for inference about coordinates of the mean of S
    # this allows us to compute observed_target, cross_cov and target_cov

    info_inv = np.linalg.inv(observed_sampler.covariance)
    target_cov = info_inv.dot(observed_sampler.covariance.dot(info_inv))[feature, feature]
    observed_target = info_inv[feature].dot(observed_sampler.center)
    cross_cov = observed_sampler.covariance.dot(info_inv[feature])

    observed_set = set(observed_set)
    if feature not in observed_set:
        raise ValueError('for full target, we can only do inference for features observed in the outcome')

    target_sd = np.sqrt(target_cov)

    def learning_proposal(sd=target_sd, center=0):
        scale = np.random.choice([0.5, 1, 1.5, 2], 1)
        return np.random.standard_normal() * sd + center

    weight_fn = learn_weights(algorithm, 
                              observed_set,
                              observed_sampler, 
                              observed_target,
                              target_cov,
                              cross_cov,
                              learning_proposal, 
                              fitter,
                              set_mode=True)

    # let's form the pivot

    target_val = np.linspace(-20 * target_set, 20 * target_sd, 5001) + observed_target
    weight_val = weight_fn(target_val) 
    exp_family = discrete_family(target_val, weight_val)  
    pivot = exp_family.cdf(hypothesis / target_cov[0, 0], x=observed_target)
    interval = exp_family.equal_tailed_interval(observed_target, alpha=alpha)
    rescaled_interval = (interval[0] * target_cov[0, 0], interval[1] * target_cov[0, 0])

    return pivot, rescaled_interval   # TODO: should do MLE as well does discrete_family do this?
