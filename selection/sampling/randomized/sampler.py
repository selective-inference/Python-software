from copy import copy

import numpy as np
import regreg.api as rr

class selective_sampler(object):

    def __init__(self, loss, 
                 linear_randomization,
                 quadratic_coef,
                 randomization, 
                 penalty,
                 solve_args={'tol':1.e-10, 'min_its':100, 'max_its':500}):

        (self.loss,
         self.linear_randomization,
         self.randomization,
         self.quadratic_coef) = (loss,
                                 linear_randomization,
                                 randomization,
                                 quadratic_coef)

        # initialize optimization problem

        self.penalty = penalty
        self.problem = rr.simple_problem(loss, penalty)

        random_term = rr.identity_quadratic(
                                quadratic_coef, 0, 
                                self.linear_randomization, 0)

        self.initial_soln = self.problem.solve(random_term,
                                               **solve_args)
        self.initial_grad = self.loss.smooth_objective(self.initial_soln, 
                                                       mode='grad')
        self.opt_vars = self.penalty.setup_sampling( \
            self.initial_grad,
            self.initial_soln,
            self.linear_randomization,
            self.quadratic_coef)

    def setup_sampling(self, data, loss_args={}):

        self.loss.setup_sampling(data, **loss_args)
        self.cur_grad = self.loss.smooth_objective(self.initial_soln, 'grad')

        self.penalty.setup_sampling(self.initial_grad,
                                    self.initial_soln,
                                    self.linear_randomization,
                                    self.quadratic_coef)

        self.state = [self.loss.data.copy(), self.opt_vars]

    def logpdf(self, state):
        """
        Log density of the randomization density.

        In the paper's notation:

        .. math::

            g(-\nabla \ell(s;\beta) - \epsilon \beta - z)

        """
        data, opt_vars = state
        param, subgrad, opt_vec = self.penalty.form_optimization_vector(opt_vars)

        self.loss.data = data
        gradient = self.loss.smooth_objective(param, 'grad')
        hessian =  self.loss.hessian(param)
        log_jacobian = self.penalty.log_jacobian(hessian)
        val = - gradient - opt_vec

        return self.randomization.logpdf(val).sum() + log_jacobian

class selective_sampler_MH(selective_sampler):

    def sampling(self,
                 ndraw=500,
                 burnin=100):
        """
        The function samples the distribution of the sufficient statistic
        subject to the selection constraints.
        """
        samples = []

        for i in range(ndraw + burnin):
            sample = self.next()
            if i >= burnin:
                samples.append(copy(sample))
        return samples

    def __iter__(self):
        return self

    def next(self):
        self.state[0] = self.loss.step_data(self.state, self.logpdf)

        # update the gradient
        param = self.penalty.form_parameters(self.state[1])
        self.loss.data = self.state[0]
        self.cur_grad = self.loss.smooth_objective(param, 'grad')

        opt_vars = self.penalty.step_variables(self.state, self.randomization, self.logpdf, self.cur_grad)
        betaE, subgrad = opt_vars

        # update the optimization variables. 
        self.state[1] = opt_vars

        return self.state

class selective_sampler_sqrtLasso(selective_sampler_MH):
    
    ### calculate the det part in sqrt lasso
    ### selected_part is X_E^T
    def abs_det(self, state):
        data, opt_vars = state
        param, subgrad, opt_vec = self.penalty.form_optimization_vector(opt_vars)
        selected_part = self.loss.X[:,self.penalty.active_set].T
        n = selected_part.shape[1]
        residual = data - np.dot(self.loss.X, param)
        R = np.identity(n) - np.outer(residual, residual) / (np.linalg.norm(residual)**2)
        temp = np.dot(selected_part, R)
        result = np.dot(temp, selected_part.T) / (np.linalg.norm(residual)**2) + self.penalty.quadratic_coef * np.identity(selected_part.shape[0])
        return np.fabs(np.linalg.det(result))
    
    def logpdf(self, state):
        """
            Log density of the randomization density plus the determinant part
            
        """
        
        data, opt_vars = state
        param, subgrad, opt_vec = self.penalty.form_optimization_vector(opt_vars)
        gradient = self.loss.gradient(data, param)
        val = - gradient - opt_vec
        generalized_logpdf = self.randomization.logpdf(val).sum() + np.log(self.abs_det(state))
        
        return generalized_logpdf

### linear_part is X_{E\j}^T
    def MH_data(self, initial):
        
        self.total_data += 1
        data, opt_vars = initial
        
    ### selected_part = self.loss.X[:,self.penalty.active_set].T
        proposal, log_transition_ratio = self.loss.proposal(data, self.loss.linear_part)
        proposal_sample = (proposal, opt_vars)
        sample = (data, opt_vars)
                                                                           
        log_ratio = (log_transition_ratio + self.logpdf(proposal_sample) - self.logpdf(sample))
                                                                           
        if np.random.uniform() < np.exp(log_ratio):
            data = proposal
            param = self.penalty.form_parameters(opt_vars)
            self.accept_data += 1
            self.gradient = self.loss.gradient(data, param)
                                                                           
        return data

