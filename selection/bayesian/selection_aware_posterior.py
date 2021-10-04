import numpy as np, sys
from selection.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from scipy.stats import norm as ndist

class posterior_inference():


    def __init__(self,
                 observed_target,
                 cov_target,
                 cov_target_score,
                 feasible_point,
                 cond_mean,
                 cond_cov,
                 logdens_linear,
                 linear_part,
                 offset,
                 ini_estimate):

        self.observed_target = observed_target
        self.cov_target = cov_target
        self.cov_target_score = cov_target_score

        self.feasible_point = feasible_point
        self.cond_mean = cond_mean
        self.cond_cov = cond_cov
        self.target_size = cond_cov.shape[0]
        self.logdens_linear = logdens_linear
        self.linear_part = linear_part
        self.offset = offset
        self.ini_estimate = ini_estimate

    def prior(self, target_parameter, var_parameter, lam):

        std_parameter = np.sqrt(var_parameter)
        grad_prior_par = -np.true_divide(target_parameter,  var_parameter)
        grad_prior_std = np.true_divide(target_parameter**2. , 2.*(var_parameter**2))- (lam/2.)-1./(2.*var_parameter)
        log_prior = -(np.linalg.norm(target_parameter)**2.) / (2.*var_parameter) - (lam * (np.linalg.norm(std_parameter)**2)/2.)-np.log(std_parameter)
        return grad_prior_par, grad_prior_std, log_prior

    def det_initial_point(self, initial_soln, solve_args={'tol':1.e-12}):

        if np.asarray(self.observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(self.observed_target)
        prec_target = np.linalg.inv(self.cov_target)

        target_lin = - self.logdens_linear.dot(self.cov_target_score.T.dot(prec_target))
        target_offset = self.cond_mean - target_lin.dot(observed_target)

        prec_opt = np.linalg.inv(self.cond_cov)
        mean_opt = target_lin.dot(initial_soln) + target_offset
        conjugate_arg = prec_opt.dot(mean_opt)

        solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        initial_point = initial_soln + self.cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
        return initial_point

    def gradient_log_likelihood(self, parameters, solve_args={'tol':1.e-15}):

        npar = self.target_size
        target_parameter = parameters[:npar]
        var_parameter = parameters[npar:]
        if np.asarray(self.observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(self.observed_target)
        prec_target = np.linalg.inv(self.cov_target)

        target_lin = - self.logdens_linear.dot(self.cov_target_score.T.dot(prec_target))
        target_offset = self.cond_mean - target_lin.dot(observed_target)

        prec_opt = np.linalg.inv(self.cond_cov)
        mean_opt = target_lin.dot(target_parameter) + target_offset
        conjugate_arg = prec_opt.dot(mean_opt)

        solver = solve_barrier_affine_C

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        reparam = target_parameter + self.cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
        neg_normalizer = (target_parameter - reparam).T.dot(prec_target).dot(target_parameter - reparam)/2. \
                         + val + mean_opt.T.dot(prec_opt).dot(mean_opt) / 2.

        grad_barrier = np.diag(2. / ((1. + soln) ** 3.) - 2. / (soln ** 3.))

        L = target_lin.T.dot(prec_opt)
        N = L.dot(hess)
        jacobian = (np.identity(observed_target.shape[0]) + self.cov_target.dot(L).dot(target_lin)) - \
                   self.cov_target.dot(N).dot(L.T)

        log_lik = -((observed_target - reparam).T.dot(prec_target).dot(observed_target - reparam)) / 2. + neg_normalizer \
                  + np.log(np.linalg.det(jacobian))

        grad_lik = jacobian.T.dot(prec_target).dot(observed_target)
        grad_neg_normalizer = -jacobian.T.dot(prec_target).dot(target_parameter)

        opt_num = self.cond_cov.shape[0]
        grad_jacobian = np.zeros(opt_num)
        A = np.linalg.inv(jacobian).dot(self.cov_target).dot(N)
        for j in range(opt_num):
            M = grad_barrier.dot(np.diag(N.T[:, j]))
            grad_jacobian[j] = np.trace(A.dot(M).dot(N.T))

        prior_info = self.hierarchical_prior(reparam, var_parameter, lam=0.01)
        return np.append(grad_lik + grad_neg_normalizer + grad_jacobian + jacobian.T.dot(prior_info[0]), prior_info[1]),\
               np.append(reparam, var_parameter), log_lik + prior_info[2]



