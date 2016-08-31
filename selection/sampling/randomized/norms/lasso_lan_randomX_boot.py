import numpy as np
#from scipy.stats import dirichlet

import regreg.api as rr
from base import selective_penalty


# same as lasso.py except uses Langevin Metropolis Hastings for simplex_step
# relies on the fact that the randomization used is Laplace to compute the \grad log \pi in the Langevin update
# Sampling from a log-concave distribution with Projected Langevin Monte Carlo (Bubeck et al)
# http://arxiv.org/pdf/1507.02564v1.pdf

# needed for adaptive MCMC
# source: git@github.com:jcrudy/choldate.git
#from choldate import cholupdate, choldowndate

## TODO: should use rr.weighted_l1norm

class selective_l1norm_lan_randomX_boot(rr.l1norm, selective_penalty):

    ### begin selective_penalty API

    ### API begins here

    def setup_sampling(self,
                       gradient,
                    #   hessian, ## added
                       soln,
                       linear_randomization,
                       quadratic_coef):

        # this will get used to randomize on simplex

        #self.simplex_randomization = (0.05, dirichlet(np.ones(self.shape)))

        self.quadratic_coef = quadratic_coef

        self.accept_l1_part, self.total_l1_part = 0, 0

        random_direction = quadratic_coef * soln + linear_randomization
        negative_subgrad = gradient + random_direction

        self.active_set = (soln != 0)
        self.inactive = (soln ==0)

        self.signs = np.sign(soln[self.active_set])

        #abs_l1part = np.fabs(soln[self.active_set])
        #l1norm_ = abs_l1part.sum()

        subgrad = -negative_subgrad[self.inactive_set] # u_{-E}
        supnorm_ = np.fabs(negative_subgrad).max()

        if self.lagrange is None:
            raise NotImplementedError("only lagrange form is implemented")

        ##TODO: replace supnorm_ with self.lagrange? check whether they are the same
        ## it seems like supnorm_ is slightly bigger than self.lagrange

        betaE, cube = soln[self.active_set], subgrad / supnorm_

        #simplex, cube = np.fabs(soln[self.active_set]), subgrad / self.lagrange

        # print cube
        # for adaptive mcmc

        nactive = soln[self.active_set].shape[0]

        self.chol_adapt = np.identity(nactive) / np.sqrt(nactive)

        #self.hessian = hessian
        return betaE, cube

    def form_subgradient(self, opt_vars):
        """
        opt_vars will be of the form returned by self.setup_sampling

        this should form z, the subgradient of P at beta

        """
        _, cube = opt_vars
        lam = self.lagrange

        active_set = self.active_set
        inactive_set = self.inactive_set

        full_subgrad = np.ones(self.shape)
        full_subgrad.flat[active_set] = self.signs
        full_subgrad.flat[inactive_set] = cube

        return lam * full_subgrad

    def form_parameters(self, opt_vars):
        """
        opt_vars will be of the form returned by self.setup_sampling

        this should form beta

        """

        betaE, _ = opt_vars

        full_params = np.zeros(self.shape)
        full_params.flat[self.active_set] = betaE

        return full_params

    def form_optimization_vector(self, opt_vars):
        """
        opt_vars will be of the form returned by self.setup_sampling

        this should form beta, z, epsilon * beta + z
        """


        # could be more efficient for LASSO

        params = self.form_parameters(opt_vars)
        subgrad = self.form_subgradient(opt_vars)
        return params, subgrad, self.quadratic_coef * params + subgrad

    def log_jacobian(self, hessian):
        if self.constant_log_jacobian is None:
            restricted_hessian = hessian[self.active_set][:,self.active_set]
            restricted_hessian += self.quadratic_coef * np.identity(restricted_hessian.shape[0])
            return np.linalg.slogdet(restricted_hessian)[1]
        else:
            return self.constant_log_jacobian

    #def step_variables(self, state, randomization, logpdf, gradient):
    #    """
    #    Updates internal parameterization of
    #    the optimization variables.

    #    """
    #    #new_cube = self.step_cube(state, randomization, gradient)

    #    data, opt_vars = state
    #    simplex, _ = opt_vars
    #    new_state = (data, (simplex, new_cube))
    #    new_simplex = self.step_simplex(new_state, randomization, logpdf, gradient, self.hessian)
    #    return new_simplex, new_cube

    ### end selective_penalty API

    ### specific to lasso

    # for asymptotically Gaussian inference
    # the LASSO's Jacobian is (asymptotically) constant

    def get_constant_log_jacobian(self):
        if hasattr(self, "_constant_log_jacobian"):
            return self._constant_log_jacobian

    def set_constant_log_jacobian(self, log_jacobian):
        self._constant_log_jacobian = log_jacobian

    constant_log_jacobian = property(get_constant_log_jacobian,
                                     set_constant_log_jacobian)

    def get_active_set(self):
        if hasattr(self, "_active_set"):
            return self._active_set

    def set_active_set(self, active_set):
        self._active_set = np.zeros(self.shape, np.bool)
        self._active_set[active_set] = 1
        self._inactive_set = ~self._active_set

        nactive = self._active_set.sum()
        ninactive = self._inactive_set.sum()

        self.dtype = np.dtype([('betaE', (np.float,    # parameters
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

    def get_penalty_params(self, scale):
        if self.lagrange is not None:
            return self.lagrange, scale
        else:
            return scale, self.bound


    def step_variables(self, state, randomization, logpdf, gradient, hessian):
        """
        Uses projected Langevin proposal ( X_{k+1} = P(X_k+\eta\grad\log\pi+\sqrt{2\pi} Z), where Z\sim\mathcal{N}(0,Id))
        for a new simplex point (a point in non-negative orthant, hence projection onto [0,\inf)^|E|)
        """
        self.total_l1_part += 1
        lam = self.lagrange

        data, opt_vars = state
        betaE, cube = opt_vars
        active = self.active_set
        inactive = ~active
        nactive = betaE.shape[0]
        ninactive = cube.shape[0]


        if self.lagrange is None:
            raise NotImplementedError("The bound form has not been implemented")


        #stepsize = 1/np.sqrt(nactive)
        stepsize = 2./float(nactive+ninactive)  # eta below

        # new for projected Langevin MCMC

        B = hessian + self.quadratic_coef*np.identity(nactive+ninactive)
        A = B[:, active]


        _grad_loglik0 = self.grad_loglik(data, opt_vars, A, hessian, gradient)
        opt_vars_proposal = self.proposal(opt_vars, stepsize, _grad_loglik0)
        betaE_proposal, cube_proposal = opt_vars_proposal



        #_grad_loglik1 = self.grad_loglik(data, opt_vars_proposal, A, hessian, _XTX_b)

        #log_ratio = logpdf((data, (betaE_proposal, cube_proposal))) - logpdf(state) \
        #            - self.q_transition(opt_vars, opt_vars_proposal, _grad_loglik0, stepsize) \
        #            + self.q_transition(opt_vars_proposal, opt_vars, _grad_loglik1, stepsize)

        # update cholesky factor

        #alpha = np.minimum(np.exp(log_ratio), 1)
        #target = 2.4 / np.sqrt(nactive)
        #multiplier = ((self.total_l1_part+1)**(-0.8) *
        #               (np.exp(log_ratio) - target))
        #rank_one = np.sqrt(np.fabs(multiplier)) * step / np.linalg.norm(random_sample)

        #if multiplier > 0:
        #     cholupdate(self.chol_adapt, rank_one) # update done in place
        #else:
        #     choldowndate(self.chol_adapt, rank_one) # update done in place

        # return proposal

        #if np.log(np.random.uniform()) < log_ratio:
        betaE, cube = betaE_proposal, cube_proposal
        self.accept_l1_part += 1

        return betaE, cube

        #return proposal

    def grad_loglik(self, data, opt_vars, A, hessian, gradient):
        params, _, opt_vec = self.form_optimization_vector(opt_vars)

        active = self.active_set
        nactive = np.sum(active)

        sign_vec = - np.sign(gradient+opt_vec)
        grad_betaE = np.dot(A.T, sign_vec)
        grad_cube = self.lagrange*sign_vec[~active]
        _grad_loglik = np.concatenate((grad_betaE, grad_cube), axis=0)
        return _grad_loglik


    def proposal(self, opt_vars, stepsize, _grad_loglik):

        betaE, cube = opt_vars
        nactive = betaE.shape[0]
        grad_betaE = _grad_loglik[:nactive]
        grad_cube = _grad_loglik[nactive:]


        betaE_proposal = betaE + stepsize*grad_betaE + np.sqrt(2*stepsize)*np.random.standard_normal(betaE.shape[0])
        cube_proposal = cube + stepsize*grad_cube + np.sqrt(2*stepsize)*np.random.standard_normal(cube.shape[0])

        #print 'betaE', betaE_proposal
        for i in range(nactive):
            if (betaE_proposal[i] * self.signs[i] < 0):
                betaE_proposal[i] = 0

        cube_proposal = np.clip(cube_proposal, -1, 1)

        return betaE_proposal, cube_proposal


    def q_transition(self, opt_vars0, opt_vars1, grad_loglik, stepsize):
        betaE0, cube0 = opt_vars0
        betaE1, cube1 = opt_vars1
        x0 = np.concatenate((betaE0, cube0), axis=0)
        x1 = np.concatenate((betaE1, cube1), axis=0)
        return -((x0-x1-stepsize*grad_loglik)**2).sum()/(4*stepsize)


