import numpy as np
import regreg.api as rr
from base import selective_penalty

class selective_l1norm(rr.l1norm, selective_penalty):

    def get_active_set(self):
        if hasattr(self, "_active_set"):
            return self._active_set

    def set_active_set(self, active_set):
        self._active_set = np.zeros(self.shape, np.bool)
        self._active_set[active_set] = 1
        self._inactive_set = ~self._active_set

        nactive = self._active_set.sum()
        ninactive = self._inactive_set.sum()

        self.dtype = np.dtype([('simplex', (np.float,    # parameters 
                                            nactive-1)), # on face simplex
                               ('scale', np.float),
                               ('signs', (np.int, nactive)),
                               ('cube', (np.float, 
                                         ninactive))])

    active_set = property(get_active_set, set_active_set, 
                          doc="The active set for selective sampling.")

    @property
    def inactive_set(self):
        if hasattr(self, "_inactive_set"):
            return self._inactive_set

    def setup_sampling(self, 
                       gradient, 
                       soln, 
                       linear_randomization,
                       quadratic_coef):

        self.accept_beta, self.total_beta = 0, 0

        random_direction = 2 * quadratic_coef * soln + linear_randomization 
        negative_subgrad = gradient + random_direction

        self.active_set = (soln != 0)
        self.initial_parameters = np.empty(1, self.dtype)
        self.initial_parameters['signs'] = np.sign(soln[self.active_set])
        abs_l1part = np.fabs(soln[self.active_set])
        l1norm_ = abs_l1part.sum()

        self.initial_parameters['simplex'] = (abs_l1part / l1norm_)[:-1]
        subgrad = -negative_subgrad[self.inactive_set]
        supnorm_ = np.fabs(negative_subgrad).max()

        if self.lagrange is not None: 
            self.initial_parameters['cube'] = subgrad / self.lagrange
            self.initial_parameters['scale'] = l1norm_
        else:
            if self._active_set.sum() != self.shape:
                self.initial_parameters['cube'] = subgrad / supnorm_
                self.initial_parameters['scale'] = supnorm_
        
        if self.lagrange is None:
            raise NotImplementedError("only lagrange form is implemented")

        return soln[self.active_set], subgrad

    def form_subgradient(self, opt_vars):

        betaE, subgrad = opt_vars
        lam = self.lagrange

        active_set = self.active_set
        inactive_set = self.inactive_set

        full_subgrad = np.ones(self.shape)
        full_subgrad.flat[active_set] = self.initial_parameters['signs']
        full_subgrad.flat[inactive_set] = subgrad

        return lam * full_subgrad

    def form_parameters(self, opt_vars):

        betaE, subgrad = opt_vars

        active_set = self.active_set
        full_beta = np.zeros(self.shape)
        full_beta.flat[active_set] = betaE

        return full_beta

    def get_penalty_params(self, scale):
        if self.lagrange is not None:
            return self.lagrange, scale
        else:
            return scale, self.bound

    ### Metropolis-Hastings steps

    def step_variables(self, state, randomization, logpdf, gradient):
        subgrad = self.step_subgrad(state, randomization, logpdf, gradient)
        betaE = self.step_beta(state, randomization, logpdf, gradient)
        return betaE, subgrad

    def step_beta(self, state, randomization, logpdf, gradient):

        self.total_beta += 1

        lam = self.lagrange
        data, opt_vars = state
        betaE, subgrad = opt_vars

        nactive = betaE.shape[0]
        stepsize = 1.3/np.sqrt(nactive)

        rand = randomization
        proposal = self.initial_parameters['signs'] * np.fabs(betaE + stepsize * rand.rvs(size=nactive)) 
        proposal_state = (data, (proposal, subgrad))

        log_ratio = (logpdf(proposal_state) - 
                     logpdf(state))

        if np.random.uniform() < np.exp(log_ratio):
            betaE = proposal
            self.accept_beta += 1
        
        return betaE

    def step_subgrad(self, state, randomization, logpdf, gradient):

        data, opt_vars = state
        betaE, subgrad = opt_vars

        lam = self.lagrange

        rand = randomization
        active_set = self.active_set
        inactive_set = self.inactive_set

        offset = - gradient[inactive_set]
        lower = offset - lam  
        upper = offset + lam  
        percentile = np.random.rand(inactive_set.sum()) \
                * (rand.cdf(upper) - rand.cdf(lower)) + rand.cdf(lower)
        subgrad_sample = (offset - rand.ppf(percentile)) / lam
        return subgrad_sample


class selective_supnorm(rr.supnorm, selective_l1norm):

    def setup_sampling(self, gradient, soln, random_direction,
                       tol=1.e-4):

        negative_subgrad = gradient + random_direction

        supnorm_ = np.fabs(soln).max()
        self.active_set = (np.fabs(soln) > (1 - tol) * supnorm_)
        self.initial_parameters = np.empty(1, self.dtype)
        self.initial_parameters['signs'] = np.sign(negative_subgrad[self.active_set])
        abs_l1part = np.fabs(negative_subgrad[self.active_set])
        l1norm_ = abs_l1part.sum()

        self.initial_parameters['simplex'] = (abs_l1part / l1norm_)[:-1]
        self.initial_parameters['cube'] = soln / supnorm_

        if self.lagrange is not None:
            self.initial_parameters['scale'] = supnorm_
        else:
            self.initial_parameters['scale'] = l1norm_

    def get_penalty_params(self, scale):
        if self.lagrange is not None:
            return self.lagrange, scale
        else:
            return scale, self.bound

