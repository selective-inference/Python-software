from copy import copy

import numpy as np
import regreg.api as rr

# I needed sampler_new since projected Langevin for step_simplex needed hessian and that was not available in penalty class
# this one has everything the same as sampler just has self.hesssian as well

class selective_sampler(object):

    def __init__(self, loss,
                 linear_randomization,  #set in e.g. tests/test_logistic_first_version.py, the model selection done based on this randomization
                 quadratic_coef,  # \epsilon in \frac{\epsilon}{2}\|\beta\|_2^2 term in the objective we are minimizing
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

        self.initial_soln = self.problem.solve(random_term,    # model selection, initial solution set here
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


'''

    def logpdf(self, state):
        """
        Log density of the randomization density, i.e. computes log of
            g(-\grad l(\beta) - \epsilon\beta - \grad P(\beta)), P - penalty, e.g. \lambda\|\beta\|_1 for the lasso
            plus log of jacobian.
            Recall the objective: min l(\beta)+P(\beta)+\frac{\epsilon}{2}\|\beta\|_2^2+w^T\beta, implies
            -w = \epsilon\beta+\grad P(\beta)+\grad l(\beta)
        """

        data, opt_vars = state   # opt_vars=(simplex, cube) in penalty class (e.g. norms\lasso.py)
        # the following is an important step that makes param as (signs*simplex, 0), subgrad = (signs, cube),
        # opt_vec = \epsilon\beta+subgrad (subgrad = \grad P(\beta), gradient of the penalty)
        # opt_vec becomes quadratic_coef*params+subgrad in penalty class
        param, subgrad, opt_vec = self.penalty.form_optimization_vector(opt_vars)
        gradient = self.loss.gradient(data, param)
        hessian =  self.loss.hessian()
        log_jacobian = self.penalty.log_jacobian(hessian)
        val = - gradient - opt_vec

        return self.randomization.logpdf(val).sum() + log_jacobian   # sum since we assume randomization is iid
'''


class selective_sampler_MH_lan(selective_sampler):

    def sampling(self,
                 ndraw=7000,
                 burnin=3000):
        """
        This function provides samples (data, \beta, subgrad) from
        normal_distribution(data)*g(gradient+epsilon*\beta+\epsilon(\beta 0))*jacobian,
        where gradient = \grad l(\beta) is a function of data and parameter \beta.
        """

        samples = []

        for i in range(ndraw + burnin):
            sample = self.next()
            if (i >= burnin):
                samples.append(copy(sample))
        return samples

    def __iter__(self):
        return self




    def next(self):
        """
        Gibbs sampler:
        calls one-step MH for the data vector (step_data in loss class), then one-step MH for simplex and moves cube vector
        (the last two done under step_variables in the penalty class)
        """

        # updates data according to MH step (might not actually move depending whether accepts or rejects)
        # step_data written in losses/base.py

        data, opt_vars = np.copy(self.state)
        param, subgrad, opt_vec = self.penalty.form_optimization_vector(opt_vars)
        gradient = self.loss.gradient(data, param)
        hessian = self.loss.hessian()

        X = self.loss.X
        P = self.loss.P
        R = self.loss.R

        p = data.shape[0]
        _full_gradient = self.penalty.full_gradient(self.state, gradient, hessian, X)
        _full_projection = self.penalty.full_projection(self.state)
        vector = np.zeros(data.shape[0]+param.shape[0]+subgrad.shape[0])
        vector[:p] = data
        vector[p:(p+param.shape[0])] = param
        vector[(p+param.shape[0]):] = subgrad

        new_state = projected_langevin(vector,
                                            _full_gradient,
                                            _full_projection,
                                            1. / p).next()

        new_data, new_opt_vars = new_state
        new_data = np.dot(P, data) + np.dot(R, new_data)

        # update the optimization variables.
        self.state[0] = new_data
        self.state[1] = new_opt_vars

        return self.state

