import numpy as np
from copy import copy
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist
import rpy2.robjects as rpy
import rpy2.robjects.numpy2ri

# description of statistical problem

truth = np.array([1,2.])

data = np.random.standard_normal((100, 2)) + np.multiply.outer((100,), truth) / np.sqrt(100)

def sufficient_stat(data):
    return np.mean(data, 0)

S = sufficient_stat(data)

# randomization mechanism

class normal_sampler(object):

    def __init__(self, center, covariance):
        (self.center,
         self.covariance) = (np.asarray(center),
                             np.asarray(covariance))
        self.cholT = np.linalg.cholesky(self.covariance)
        self.shape = self.center.shape

    def __call__(self, scale=1., size=None):

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        if self.shape == ():
            _shape = (1,)
        else:
            _shape = self.shape
        return np.sqrt(scale) * np.squeeze(np.random.standard_normal(size + _shape).dot(self.cholT))

    def __copy__(self):
        return normal_sampler(self.center.copy(),
                              self.covariance.copy())

observed_sampler = normal_sampler(S, 1/100. * np.identity(2))   

assert(observed_sampler(size=(200,)).shape == (200, 2)) # 200 iid draws N(S, sigma) (given S)

def algo_constructor():

    def myalgo(sampler):
        noisyS = sampler(scale=0.25)
        return np.fabs(noisyS.sum()) > 2

    return myalgo

# run selection algorithm

algo_instance = algo_constructor()
observed_outcome = algo_instance(observed_sampler)

# find the target, based on the observed outcome

def compute_target(observed_outcome, data):
    if observed_outcome: # target is truth[0]
        observed_target, target_cov, cross_cov = sufficient_stat(data)[0]/100, 1/100. * np.identity(1), np.array([1., 0.]).reshape((2,1))
    else:
        observed_target, target_cov, cross_cov = sufficient_stat(data)[1]/100, 1/100. * np.identity(1), np.array([0., 1.]).reshape((2,1))
    return observed_target, target_cov, cross_cov

observed_target, target_cov, cross_cov = compute_target(observed_outcome, data)
direction = cross_cov.dot(np.linalg.inv(target_cov))

def learning_proposal():
    scale = np.random.choice([0.5, 1, 2, 3], 1)
    return np.random.standard_normal() * scale

def logit_fit(T, Y):
    rpy2.robjects.numpy2ri.activate()
    rpy.r.assign('T', T)
    rpy.r.assign('Y', Y.astype(np.int))
    rpy.r('''
    Y = as.numeric(Y)
    T = as.numeric(T)
    M = glm(Y ~ T, family=binomial(link='logit'))
    coefM = coef(M)
    fitfn = function(t) { linpred = coefM[1] + t * coefM[2]; return(linpred) }
    ''')
    rpy2.robjects.numpy2ri.deactivate()

    def fitfn(t):
        rpy2.robjects.numpy2ri.activate()
        fitfn_r = rpy.r('fitfn')
        val = fitfn_r(t)
        rpy2.robjects.numpy2ri.deactivate()
        return np.exp(val) / (1 + np.exp(val))

    return fitfn

def probit_fit(T, Y):
    rpy2.robjects.numpy2ri.activate()
    rpy.r.assign('T', T)
    rpy.r.assign('Y', Y.astype(np.int))
    rpy.r('''
    Y = as.numeric(Y)
    T = as.numeric(T)
    M = glm(Y ~ T, family=binomial(link='probit'))
    coefM = coef(M)
    fitfn = function(t) { linpred = coefM[1] + t * coefM[2]; return(linpred) }
    ''')
    rpy2.robjects.numpy2ri.deactivate()

    def fitfn(t):
        rpy2.robjects.numpy2ri.activate()
        fitfn_r = rpy.r('fitfn')
        val = fitfn_r(t)
        rpy2.robjects.numpy2ri.deactivate()
        return ndist.cdf(val)

    return fitfn

def learn_weights(algorithm, observed_sampler, learning_proposal, fit_probability, B=15000):

    new_sampler = copy(observed_sampler)

    learning_sample = []
    for _ in range(1000):
         T = learning_proposal()      # a guess at informative distribution for learning what we want
         new_sampler.center = S + direction * (T - observed_target)
         Y = algorithm(new_sampler) == observed_outcome
         learning_sample.append((T, Y))
    learning_sample = np.array(learning_sample)
    T, Y = learning_sample.T
    conditional_law = fit_probability(T, Y)
    return conditional_law

weight_fn = learn_weights(algo_instance, observed_sampler, learning_proposal, logit_fit)

# let's form the pivot

if observed_outcome:
    true_target = truth[0]
else:
    true_target = truth[1]

target_val =  np.linspace(-1,1,2001)
weight_val = weight_fn(target_val) * ndist.pdf(target_val / np.sqrt(target_cov[0,0]))

# for p == 1 targets this is what we do -- have some code for multidimensional too

exp_family = discrete_family(target_val, weight_val)  
pivot = exp_family.cdf(true_target, observed_target)
interval = exp_family.equal_tailed_interval(observed_target) 
