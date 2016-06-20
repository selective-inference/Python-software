from copy import copy

import numpy as np
import regreg.api as rr

class kac_rice_sampler():

    def __init__(self,
                 quadratic_coef,
                 randomization):

        (self.randomization,
         self.quadratic_coef) = (randomization,
                                 quadratic_coef)

        # initialize optimization problem



    def setup_sampling(self, X, data, eta, subgrad, A, b):

        self.X = X
        self.total = 0
        self.accept = 0
        self.state = [data.copy(), eta.copy(), subgrad.copy()]

        self.A = A
        self.b = b
        self.rho = 0.5
        self.mat = np.linalg.inv(np.identity(self.A.shape[1])+self.rho*np.dot(self.A.T, self.A))
        self.it = 100

    def admm_update(self, subgrad, u, r, x):

        u_new = np.dot(self.mat, subgrad+self.rho*np.dot(self.A.T,self.b+r-x))
        r_new = np.minimum(np.dot(self.A, u_new)-self.b+x,0)
        x_new = x+np.dot(self.A,u_new)-self.b-r_new

        return u_new, r_new, x_new

    def admm_projection(self, subgrad):

        nrow_A = self.A.shape[0]
        u = subgrad.copy()
        r = np.minimum(np.dot(self.A, u)-self.b,0)
        x = np.zeros(nrow_A)
        for i in range(self.it):
            u, r, x = self.admm_update(subgrad, u, r, x)

        return u


    def step_variables(self):
        """
        """
        self.total += 1

        data, eta, subgrad = self.state

        p = eta.shape[0]
        stepsize = 1 / float(p)


        sign_vec = np.sign(-np.dot(self.X.T, data) + subgrad - self.quadratic_coef * eta)

        grad_eta_loglik = self.quadratic_coef * sign_vec
        eta_proposal = eta + (stepsize * grad_eta_loglik) + (np.sqrt(2 * stepsize) * np.random.standard_normal(eta.shape[0]))


        grad_subgrad_loglik = - sign_vec
        subgrad_proposal = subgrad + (stepsize * grad_subgrad_loglik) + (np.sqrt(2 * stepsize) * np.random.standard_normal(subgrad.shape[0]))

        subgrad_proposal = self.admm_projection(subgrad_proposal)
        ## projection of the subgradient MISSING

        grad_y_loglik = - (data - np.dot(self.X, sign_vec))
        data_proposal = data + (stepsize * grad_y_loglik) + (np.sqrt(2 * stepsize) * np.random.standard_normal(data.shape[0]))

        data, eta, subgrad = data_proposal, eta_proposal, subgrad_proposal
        self.accept += 1

        self.state = [data, eta, subgrad]
        return data, eta, subgrad



    def sampling(self,
                 ndraw=5000,
                 burnin=1000):
        """
        The function samples the distribution of the sufficient statistic
        subject to the selection constraints.
        """
        samples = []

        for i in range(ndraw + burnin):
            sample = self.step_variables()
            if i >= burnin:
                samples.append(copy(sample))
        return samples


