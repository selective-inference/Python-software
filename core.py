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

    def __call__(self, size=None, scale=1.):

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        if self.shape == ():
            _shape = (1,)
        else:
            _shape = self.shape
        return scale * np.squeeze(np.random.standard_normal(size + _shape).dot(self.cholT)) + self.center

    def __copy__(self):
        return self.__class__(self.center.copy(),
                              self.covariance.copy())

class split_sampler(object):

    def __init__(self, sample_stat, covariance): # covariance of sum of rows
        self.sample_stat = np.asarray(sample_stat)
        self.nsample = self.sample_stat.shape[0]
        self.center = np.sum(self.sample_stat, 0)
        self.covariance = covariance
        self.shape = self.center.shape

    def __call__(self, size=None, scale=0.5):

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

    def __copy__(self):
        return split_sampler(self.stat_sample.copy(),
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
                  check_selection=None):
    """
    The algorithm to learn P(Y=1|T)
    """
    S = selection_stat = observed_sampler.center

    new_sampler = normal_sampler(observed_sampler.center.copy(),
                                 observed_sampler.covariance.copy())

    if check_selection is None:
        check_selection = lambda result: result == observed_outcome

    direction = cross_cov.dot(np.linalg.inv(target_cov).reshape((1,1))) # move along a ray through S with this direction

    learning_Y, learning_T = [], []
    for _ in range(B):
         T = learning_proposal()      # a guess at informative distribution for learning what we want
         new_sampler.center = S + direction.dot(T - observed_target)
         new_result = algorithm(new_sampler)

         Y = check_selection(new_result)

         learning_Y.append(Y)
         learning_T.append(T)

    learning_Y = np.array(learning_Y, np.float)
    learning_T = np.squeeze(np.array(learning_T, np.float))

    conditional_law = fit_probability(learning_T, learning_Y)
    return conditional_law

def infer_general_target(algorithm,
                         observed_outcome,
                         observed_sampler,
                         observed_target,
                         cross_cov,
                         target_cov,
                         fitter=logit_fit,
                         hypothesis=0,
                         alpha=0.1,
                         B=15000):

    target_sd = np.sqrt(target_cov[0, 0])

    def learning_proposal(sd=target_sd, center=observed_target):
        scale = np.random.choice([0.5, 1, 1.5, 2], 1)
        return np.random.standard_normal() * sd * scale + center

    weight_fn = learn_weights(algorithm, 
                              observed_outcome,
                              observed_sampler, 
                              observed_target,
                              target_cov,
                              cross_cov,
                              learning_proposal, 
                              fitter,
                              B=B)

    return _inference(observed_target,
                      target_cov,
                      weight_fn,
                      hypothesis=hypothesis,
                      alpha=alpha)

def infer_full_target(algorithm,
                      observed_set,
                      feature,
                      observed_sampler,
                      dispersion, # sigma^2
                      fitter=logit_fit,
                      hypothesis=0,
                      alpha=0.1,
                      B=15000):

    # this makes assumption that covariance in observed sampler is the 
    # true covariance of S
    # and we are looking for inference about coordinates of the mean of S
    # this allows us to compute observed_target, cross_cov and target_cov

    # seems to be missing dispersion

    info_inv = np.linalg.inv(observed_sampler.covariance / dispersion) # scale free, i.e. X.T.dot(X) without sigma^2
    target_cov = (info_inv[feature, feature] * dispersion).reshape((1, 1))
    observed_target = np.squeeze(info_inv[feature].dot(observed_sampler.center))
    cross_cov = observed_sampler.covariance.dot(info_inv[feature]).reshape((-1,1))

    observed_set = set(observed_set)
    if feature not in observed_set:
        raise ValueError('for full target, we can only do inference for features observed in the outcome')

    target_sd = np.sqrt(target_cov[0, 0])

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
                              fitter,
                              check_selection=lambda result: feature in set(result),
                              B=B)

    return _inference(observed_target,
                      target_cov,
                      weight_fn,
                      hypothesis=hypothesis,
                      alpha=alpha)

def _inference(observed_target,
               target_cov,
               weight_fn, # our fitted function
               hypothesis=0,
               alpha=0.1):

    target_sd = np.sqrt(target_cov[0, 0])
              
    target_val = np.linspace(-20 * target_sd, 20 * target_sd, 5001) + observed_target
    weight_val = weight_fn(target_val) 
    weight_val *= ndist.pdf(target_val / target_sd)
    exp_family = discrete_family(target_val, weight_val)  

    pivot = exp_family.cdf(hypothesis / target_cov[0, 0], x=observed_target)

    interval = exp_family.equal_tailed_interval(observed_target, alpha=alpha)
    rescaled_interval = (interval[0] * target_cov[0, 0], interval[1] * target_cov[0, 0])

    return pivot, rescaled_interval   # TODO: should do MLE as well does discrete_family do this?
