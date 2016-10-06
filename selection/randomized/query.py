import numpy as np
import regreg.api as rr

class query(object):

    def __init__(self, randomization):

        self.randomization = randomization
        self._solved = False
        self._randomized = False

    # Methods reused by subclasses
         
    def randomize(self):

        if not self._randomized:
            self.randomized_loss = self.randomization.randomize(self.loss, self.epsilon)
        self._randomized = True

    def randomization_gradient(self, data_state, data_transform, opt_state):
        """
        Randomization derivative at full state.
        """

        if not hasattr(self, "opt_transform"):
            raise ValueError('setup_sampler should be called before using this function')

        # reconstruction of randoimzation omega

        opt_linear, opt_offset = self.opt_transform
        data_linear, data_offset = data_transform
        data_piece = data_linear.dot(data_state) + data_offset
        opt_piece = opt_linear.dot(opt_state) + opt_offset

        # value of the randomization omega

        full_state = (data_piece + opt_piece) 

        # gradient of negative log density of randomization at omega

        randomization_derivative = self.randomization.gradient(full_state)

        # chain rule for data, optimization parts

        data_grad = data_linear.T.dot(randomization_derivative)
        opt_grad = opt_linear.T.dot(randomization_derivative)

        return data_grad, opt_grad - self.grad_log_jacobian(opt_state)

    def linear_decomposition(self, target_score_cov, target_cov, observed_target_state):
        """
        Compute out the linear decomposition
        of the score based on the target. This decomposition
        writes the (limiting CLT version) of the data in the score as linear in the 
        target and in some independent Gaussian error.
        
        This second independent piece is conditioned on, resulting
        in a reconstruction of the score as an affine function of the target
        where the offset is the part related to this independent
        Gaussian error.
        """

        target_score_cov = np.atleast_2d(target_score_cov) 
        target_cov = np.atleast_2d(target_cov) 
        observed_target_state = np.atleast_1d(observed_target_state)

        linear_part = target_score_cov.T.dot(np.linalg.pinv(target_cov))

        offset = self.observed_score_state - linear_part.dot(observed_target_state)

        # now compute the composition of this map with
        # self.score_transform

        score_linear, score_offset = self.score_transform
        composition_linear_part = score_linear.dot(linear_part)

        composition_offset = score_linear.dot(offset) + score_offset

        return (composition_linear_part, composition_offset)

    # Abstract methods to be
    # implemented by subclasses

    def grad_log_jacobian(self, opt_state):
        """
        log_jacobian depends only on data through
        Hessian at \bar{\beta}_E which we 
        assume is close to Hessian at \bar{\beta}_E^*
        """
        # needs to be implemented for group lasso
        return 0.

    def solve(self):

        raise NotImplementedError('abstract method')

    def setup_sampler(self):
        """

        Setup query to prepare for sampling.

        Should set a few key attributes:
        
            - observed_score_state
            - num_opt_var
            - observed_opt_state
            - opt_transform
            - score_transform
            
        """
        raise NotImplementedError('abstract method -- only keyword arguments')

    def projection(self, opt_state):

        raise NotImplementedError('abstract method -- projection of optimization variables')

