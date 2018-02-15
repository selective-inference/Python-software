import functools
import numpy as np
import regreg.api as rr

from .query import query, optimization_sampler
from .base import restricted_estimator

class greedy_score_step(query):

    def __init__(self, 
                 loss, 
                 penalty, 
                 active_groups, 
                 candidate_groups, 
                 randomization, 
                 solve_args={'min_its':50, 'tol':1.e-10},
                 beta_active=None):
        """
        penalty is a group_lasso object that assigns weights to groups
        """

        query.__init__(self, randomization)

        (self.loss,
         self.penalty,
         self.active_groups,
         self.candidate_groups,
         self.randomization,
         self.solve_args,
         self.beta_active) = (loss,
                              penalty,
                              active_groups,
                              candidate_groups,
                              randomization,
                              solve_args,
                              beta_active)
         
        self.active = np.zeros(self.loss.shape, np.bool)
        self.candidate = np.zeros(self.loss.shape, np.bool)
        for i, g in enumerate(np.unique(self.penalty.groups)):
            if self.active_groups[i]:
                self.active[self.penalty.groups == g] = True
            elif self.candidate_groups[i]:
                self.candidate[self.penalty.groups == g] = True
                
        # we form a dual group lasso object
        # to compute the max score

        new_groups = penalty.groups[self.candidate]
        new_weights = dict([(g,penalty.weights[g]) for g in penalty.weights.keys() if g in np.unique(new_groups)])

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, lagrange=1.)

    def solve(self, nboot=2000):

        (loss,
         penalty,
         active,
         candidate,
         randomization,
         solve_args,
         beta_active) = (self.loss,
                         self.penalty,
                         self.active,
                         self.candidate,
                         self.randomization,
                         self.solve_args,
                         self.beta_active)

        if beta_active is None:
            beta_active = self.beta_active = restricted_estimator(self.loss, active, solve_args=solve_args)
            
        beta_full = np.zeros(loss.shape)
        beta_full[active] = beta_active
            
        # score at unpenalized M-estimator

        self.observed_internal_state = self.observed_score_state = - self.loss.smooth_objective(beta_full, 'grad')[candidate]
        self._randomZ = self.randomization.sample()

        self.num_opt_var = self._randomZ.shape[0]

        # find the randomized maximizer

        randomized_score = self.observed_score_state - self._randomZ
        terms = self.group_lasso_dual.terms(randomized_score)

        # assuming a.s. unique maximizing group here

        maximizing_group = np.unique(self.group_lasso_dual.groups)[np.argmax(terms)]
        maximizing_subgrad = self.observed_score_state[self.group_lasso_dual.groups == maximizing_group]
        maximizing_subgrad /= np.linalg.norm(maximizing_subgrad) # this is now a unit vector
        maximizing_subgrad *= self.group_lasso_dual.weights[maximizing_group] # now a vector of length given by weight of maximizing group
        self.maximizing_subgrad = np.zeros(candidate.sum())
        self.maximizing_subgrad[self.group_lasso_dual.groups == maximizing_group] = maximizing_subgrad
        self.observed_scaling = np.max(terms) / self.group_lasso_dual.weights[maximizing_group]

        # which groups did not win

        losing_groups = [g for g in np.unique(self.group_lasso_dual.groups) if g != maximizing_group]
        losing_set = np.zeros_like(self.maximizing_subgrad, np.bool)
        for g in losing_groups:
            losing_set += self.group_lasso_dual.groups == g

        # (candidate_subgradients, scaling) are in this epigraph:
        losing_weights = dict([(g, self.group_lasso_dual.weights[g]) for g in self.group_lasso_dual.weights.keys() if g in losing_groups])
        self.group_lasso_dual_epigraph = rr.group_lasso_dual_epigraph(self.group_lasso_dual.groups[losing_set], weights=losing_weights)
        
        self.observed_subgradients = -randomized_score[losing_set]
        self.losing_padding_map = np.identity(losing_set.shape[0])[:,losing_set]

        # which variables are added to the model

        winning_variables = self.group_lasso_dual.groups == maximizing_group
        padding_map = np.identity(self.active.shape[0])[:,self.candidate]
        self.maximizing_variables = padding_map.dot(winning_variables) > 0
        
        self.selection_variable = {'maximizing_group':maximizing_group, 
                                   'maximizing_direction':self.maximizing_subgrad,
                                   'variables':self.maximizing_variables}

        # need to implement Jacobian
        self.nboot = nboot
        self.ndim = self.loss.shape[0]

        # setup opt state and transforms

        self.observed_opt_state = np.hstack([self.observed_subgradients,
                                             self.observed_scaling])

        p = self.candidate.sum() # shorthand
        _opt_linear_term = np.zeros((p, 1 + self.observed_subgradients.shape[0]))
        _opt_linear_term[:,:self.observed_subgradients.shape[0]] = self.losing_padding_map
        _opt_linear_term[:,-1] = self.maximizing_subgrad

        _score_linear_term = np.identity(p)

        self.opt_transform = (_opt_linear_term, np.zeros(_opt_linear_term.shape[0]))
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self._solved = True
        self._setup = True


    def setup_sampler(self):
        pass

    def get_sampler(self):
        # now setup optimization sampler

        if not hasattr(self, "_sampler"):
            def projection(epigraph, opt_state):
                """
                Full projection for Langevin.

                The state here will be only the state of the optimization variables.
                """
                return epigraph.cone_prox(opt_state)
            projection = functools.partial(projection, self.group_lasso_dual_epigraph)

            def grad_log_density(query,
                                 rand_gradient,
                                 score_state,
                                 opt_state):
                full_state = score_state + reconstruct_opt(query.opt_transform, opt_state)
                return opt_linear.T.dot(rand_gradient(full_state))

            grad_log_density = functools.partial(grad_log_density, self, self.randomization.gradient)

            def log_density(query,
                            opt_linear,
                            rand_log_density,
                            score_state,
                            opt_state):
                full_state = score_state + reconstruct_opt(query.opt_transform, opt_state)
                return rand_log_density(full_state)
            log_density = functools.partial(log_density, self, self.randomization.log_density)

            self._sampler = optimization_sampler(self.observed_opt_state,
                                                 self.observed_score_state,
                                                 self.score_transform,
                                                 self.opt_transform,
                                                 projection,
                                                 grad_log_density,
                                                 log_density)
        return self._sampler

    sampler = property(get_sampler, query.set_sampler)
