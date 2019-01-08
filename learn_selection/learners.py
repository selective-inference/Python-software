import functools

import numpy as np
from scipy.stats import norm as ndist

from selection.distributions.discrete_family import discrete_family

from .samplers import normal_sampler

class mixture_learner(object):

    scales = [0.5, 1, 1.5, 2, 5, 10] # for drawing noise

    def __init__(self,
                 algorithm, 
                 observed_outcome,
                 observed_sampler, 
                 observed_target,
                 target_cov,
                 cross_cov):

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
        """


        (self.algorithm,
         self.observed_outcome,
         self.observed_sampler,
         self.observed_target,
         self.target_cov,
         self.cross_cov) = (algorithm,
                            observed_outcome,
                            observed_sampler,
                            observed_target,
                            target_cov,
                            cross_cov)

        self._chol = np.linalg.cholesky(self.target_cov)
        self._cholinv = np.linalg.inv(self._chol)

    def learning_proposal(self):
        center = self.observed_target
        scale = np.random.choice(self.scales, 1)
        return self._chol.dot(np.random.standard_normal(center.shape)) * scale + center                    

    def proposal_density(self, target_val):
        '''
        The (conditional, given self.center) density of our draws.

        Parameters
        ----------

        target_val : np.ndarray((-1, self.center.shape))

        '''

        target_val = np.asarray(target_val)
        if target_val.ndim != 2:
            raise ValueError('target_val should be 2-dimensional -- otherwise possibly ambiguous')
        center = self.observed_target
        Z = (target_val - center[None, :]).dot(self._cholinv.T)
        arg = (Z**2).sum(1) / 2.
        return np.array([np.exp(-arg/scale**2) for scale in self.scales]).mean(0)

    def generate_data(self,
                      B=500,
                      check_selection=None):

        """

        Parameters
        ----------

        B : int
            How many queries?

        check_selection : callable (optional)
            Callable that determines selection variable.

        Returns
        -------

        Y : np.array((B, -1))
            Binary responses for learning selection.

        T : np.array((B, -1))
            Points of targets where reponse evaluated -
            features in learning algorithm. Successive
            draws from `self.learning_proposal`.
        
        algorithm : callable
            Algorithm taking arguments of shape (T.shape[1],) --
            returns something of shape (Y.shape[1],).

        """

        (algorithm,
         observed_outcome,
         observed_sampler,
         observed_target,
         target_cov,
         cross_cov) = (self.algorithm,
                       self.observed_outcome,
                       self.observed_sampler,
                       self.observed_target,
                       self.target_cov,
                       self.cross_cov) 

        S = selection_stat = observed_sampler.center

        new_sampler = normal_sampler(observed_sampler.center.copy(),
                                     observed_sampler.covariance.copy())

        if check_selection is None:
            def check_selection(result):
                return [result == observed_outcome]

        direction = cross_cov.dot(np.linalg.inv(target_cov)) # move along a ray through S with this direction

        learning_Y, learning_T = [], []

        def random_meta_algorithm(new_sampler, algorithm, check_selection, T):
             new_sampler.center = S + direction.dot(T - observed_target)
             new_result = algorithm(new_sampler)
             return check_selection(new_result)

        random_algorithm = functools.partial(random_meta_algorithm, new_sampler, algorithm, check_selection)

        # this is the "active learning bit"
        # START

        for _ in range(B):
             T = self.learning_proposal()      # a guess at informative distribution for learning what we want
             Y = random_algorithm(T)

             learning_Y.append(Y)
             learning_T.append(T)

        learning_Y = np.array(learning_Y, np.float)
        learning_T = np.array(learning_T, np.float)
        if observed_target.shape == ():
            learning_Y.reshape((-1, 1))
            learning_T.reshape((-1, 1))

        return learning_Y, learning_T, random_algorithm

    def learn(self,
              fit_probability,
              fit_args = {},
              B=500,
              check_selection=None):
                  
        """
        fit_probability : callable
            Function to learn a probability model P(Y=1|T) based on [T, Y].

        fit_args : dict
            Keyword arguments to `fit_probability`.

        B : int
            How many queries?

        check_selection : callable (optional)
            Callable that determines selection variable.

        """

        learning_Y, learning_T, random_algorithm = self.generate_data(B=B,
                                                                      check_selection=check_selection)
        print('prob(select): ', np.mean(learning_Y, 0))
        conditional_laws = fit_probability(learning_T, learning_Y, **fit_args)
        return conditional_laws, (learning_T, learning_Y)

class sparse_mixture_learner(mixture_learner):

    """
    Move only along one dimension at a time
    """

    def learning_proposal(self):
        center = self.observed_target
        scale = np.random.choice(self.scales, 1)
        idx = np.random.choice(np.arange(center.shape[0]))
        prop = center.copy()
        prop[idx] = prop[idx] + np.sqrt(self.target_cov[idx, idx]) * np.random.standard_normal() * scale
        return prop + self._chol.dot(np.random.standard_normal(center.shape)) * 0.

    def proposal_density(self, target_val):
        raise NotImplementedError
