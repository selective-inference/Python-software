import functools

import numpy as np
from scipy.stats import norm as ndist

from selection.distributions.discrete_family import discrete_family

from samplers import normal_sampler

class mixture_learner(object):

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

    def learning_proposal(self):
        sd = np.sqrt(self.target_cov[0, 0])
        center = self.observed_target
        scale = np.random.choice([0.5, 1, 1.5, 2], 1)
        return np.random.standard_normal() * sd * scale + center                    

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
                return result == observed_outcome

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
             T = self.learning_proposal()      # a guess at informative distribution for learning what we want
             Y = random_algorithm(T)

             learning_Y.append(Y)
             learning_T.append(T)

        learning_Y = np.array(learning_Y, np.float)
        learning_T = np.squeeze(np.array(learning_T, np.float))

        print('prob(select): ', np.mean(learning_Y))
        conditional_law = fit_probability(learning_T, learning_Y, **fit_args)
        return conditional_law

