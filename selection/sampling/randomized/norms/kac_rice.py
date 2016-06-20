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
from choldate import cholupdate, choldowndate

## TODO: should use rr.weighted_l1norm

class selective_kac_rice(rr.l1norm, selective_penalty):



    def setup_sampling(self,
                       linear_randomization,
                       quadratic_coef):

        # this will get used to randomize on simplex

        #self.simplex_randomization = (0.05, dirichlet(np.ones(self.shape)))
        self.quadratic_coef = quadratic_coef

        self.accept_l1_part, self.total_l1_part = 0, 0

        random_direction = quadratic_coef * soln + linear_randomization
        negative_subgrad = gradient + random_direction

        self.active_set = (soln != 0)

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


        return betaE, cube






    def step_variables(self, state, randomization, X):
        """
        """
        self.total_l1_part += 1

        data, eta, subgrad = state

        p = eta.shape[0]
        stepsize = 1/float(p)


        _ , _ , opt_vec = self.form_optimization_vector(opt_vars) # opt_vec=\epsilon(\beta 0)+u, u=\grad P(\beta), P penalty


        sign_vec = np.sign(-np.dot(X.T, data)+subgrad-self.quadratic_coef*eta)


        grad_eta_loglik =  self.quadratic_coef*sign_vec


        eta_proposal = eta+(stepsize*grad_log_pi)+(np.sqrt(2*stepsize)*np.random.standard_normal(nactive))


        grad_subgrad_loglik = - sign_vec
        subgrad_proposal = subgrad + (stepsize*grad_subgrad_loglik_)+(np.sqrt(2*stepsize)*np.random.standard_normal(subgrad.shape[0]))

        ## projection of the subgradient MISSING

        grad_y_loglik = - (data - np.dot(X, sign_vec))
        data_proposal = data + (stepsize*grad_y_loglik)+(np.sqrt(2*stepsize)*np.random.standard_normal(data.shape[0]))


        
        data, eta, subgrad = data_proposal, eta_proposal, subgrad_proposal
        self.accept_l1_part += 1

        return data, eta, subgrad

