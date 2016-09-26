import numpy as np
from .multiple_views import targeted_sampler
from ..sampling.langevin import projected_langevin
class bootstrapped_target_sampler(targeted_sampler):

    # make one of these for each hypothesis test

    def __init__(self,
                 multi_view,
                 target_bootstrap,
                 observed_target_state,
                 boot_size,
                 target_alpha,
                 target_set=None,
                 reference=None):

        # sampler will draw bootstrapped weights for the target

        targeted_sampler.__init__(self, multi_view,
                                  target_bootstrap,
                                  observed_target_state,
                                  target_set,
                                  reference)
        # for bootstrap
        self.boot_size = boot_size
        self.target_alpha = target_alpha
        self.boot_transform = []
        #self.inv_mat = np.linalg.inv(np.dot(self.target_alpha, self.target_alpha.T))


        for i in range(self.nviews):
            composition_linear_part, composition_offset = self.objectives[i].linear_decomposition(self.score_cov[i],
                                                                                 self.target_cov,
                                                                                 self.observed_target_state)
            boot_linear_part = np.dot(composition_linear_part, target_alpha)
            boot_offset = composition_offset - np.dot(composition_linear_part, self.reference).flatten()
            self.boot_transform.append((boot_linear_part, boot_offset))


        self.reference_inv = self.target_inv_cov.dot(self.reference)

        # set the observed state for bootstrap

        self.boot_slice = slice(multi_view.num_opt_var, multi_view.num_opt_var + self.boot_size)
        self.observed_state = np.zeros(multi_view.num_opt_var + self.boot_size)
        self.observed_state[self.boot_slice] = np.ones(self.boot_size)
        #self.boot_observed_state[self.boot_slice] = np.random.normal(size=self.boot_size)
        self.observed_state[self.overall_opt_slice] = multi_view.observed_opt_state

        self.keep_slice = self.boot_slice
        #self.gradient = self.boot_gradient

    def gradient(self, state):

        boot_state, opt_state = state[self.boot_slice], state[self.overall_opt_slice]
        boot_grad, opt_grad = np.zeros_like(boot_state), np.zeros_like(opt_state)
        full_grad = np.zeros_like(state)

        # randomization_gradient are gradients of a CONVEX function

        for i in range(self.nviews):
            boot_grad_curr, opt_grad[self.opt_slice[i]] = \
                self.objectives[i].randomization_gradient(boot_state, self.boot_transform[i],
                                                          opt_state[self.opt_slice[i]])
            boot_grad += boot_grad_curr.copy()

        boot_grad = -boot_grad
        boot_grad -= boot_state
        #boot_grad -= np.dot(np.dot(self.target_alpha.T, self.inv_mat), self.target_alpha.dot(boot_state))
        #boot_grad -= np.dot(np.dot(self.target_alpha.T, self.target_inv_cov), self.target_alpha.dot(boot_state))

        full_grad[self.boot_slice] = boot_grad
        full_grad[self.overall_opt_slice] = -opt_grad

        return full_grad
