"""
Classes encapsulating some common workflows in randomized setting
"""

from copy import copy
import functools

import numpy as np
import regreg.api as rr
from .lasso import highdim
#from .randomization import randomization
from .query import multiple_queries, optimization_intervals, affine_gaussian_sampler
#from .M_estimator import restricted_Mest
from .base import restricted_estimator
from ..constraints.affine import constraints
from scipy.optimize import bisect

class lasso_iv(highdim):

    r"""
    A class for the LASSO with invalid instrumental variables for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\alpha, \beta} \frac{1}{2} \|P_Z (y-Z\alpha-D\beta)\|^2_2 + 
            \lambda \|\alpha\|_1 - \omega^T(\alpha \beta) + \frac{\epsilon}{2} \|(\alpha \beta)\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty.

    NOTE: use beta_tsls instead of the tsls test statistic itself, such to better fit the package structure

    """

    # add .Z field for lasso IV subclass
    def __init__(self,
                 Y, 
                 D,
                 Z, 
                 penalty=None, 
                 ridge_term=None,
                 randomizer_scale=None):
    
        # form the projected design and response
        P_Z = Z.dot(np.linalg.pinv(Z))
        X = np.hstack([Z, D.reshape((-1,1))])
        P_ZX = P_Z.dot(X)
        P_ZY = P_Z.dot(Y)
        loglike = rr.glm.gaussian(P_ZX, P_ZY)

        n, p = Z.shape

        if penalty is None:
            penalty = 2.01 * np.sqrt(n * np.log(n))
        penalty = np.ones(loglike.shape[0]) * penalty
        penalty[-1] = 0.

        mean_diag = np.mean((P_ZX**2).sum(0))
        if ridge_term is None:
            #ridge_term = 1. * np.sqrt(n)
            ridge_term = (np.std(P_ZY) * np.sqrt(mean_diag) / np.sqrt(n - 1.))

        if randomizer_scale is None:
            #randomizer_scale = 0.5*np.sqrt(n)
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(P_ZY) * np.sqrt(n / (n - 1.))

        highdim.__init__(self, loglike, penalty, ridge_term, randomizer_scale)
        self.Z = Z
        self.D = D
        self.Y = Y

        #self.sampler = affine_gaussian_test_stat_sampler(affine_con,
        #                                                self.observed_opt_state,
        #                                                self.observed_score_state,
        #                                                log_density,
        #                                                logdens_transform,
        #                                                selection_info=self.selection_variable)


    def summary(self,
                parameter=None,
                Sigma_11=1.,
                level=0.95,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=True):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter beta_star -- defaults to 0.

        Sigma_11 : true Sigma_11, known for now

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = np.zeros(1)

        parameter = np.atleast_1d(parameter)

        # compute tsls, i.e. the observed_target

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        #P_ZX, P_ZY = self.loglike.data
        #P_ZD = P_ZX[:,-1]
        #two_stage_ls = (P_ZD.dot(P_Z-P_ZE).dot(self.P_ZY-P_ZD*parameter))/np.sqrt(Sigma_11*P_ZD.dot(P_Z-P_ZE).dot(P_ZD))
        #denom = P_ZD.dot(P_Z - P_ZE).dot(P_ZD)
        #two_stage_ls = (P_ZD.dot(P_Z - P_ZE).dot(P_ZY)) / denom

        denom = self.D.dot(P_Z - P_ZE).dot(self.D)
        two_stage_ls = self.D.dot(P_Z - P_ZE).dot(self.Y) / denom

        two_stage_ls = np.atleast_1d(two_stage_ls)
        observed_target = two_stage_ls

        # only has the parametric version right now
        # compute cov_target, cov_target_score

        cov_target = np.atleast_2d(Sigma_11/denom)
        #score_cov = -1.*np.sqrt(Sigma_11/P_ZD.dot(P_Z-P_ZE).dot(P_ZD))*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        #cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.Z.T.dot(P_Z - P_ZE).dot(self.D), self.D.dot(P_Z - P_ZE).dot(self.D)])
        cov_target_score = np.atleast_2d(cov_target_score)

        alternatives = ['twosided']

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target, 
                                                  cov_target, 
                                                  cov_target_score, 
                                                  parameter=parameter, 
                                                  sample=opt_sample,
                                                  alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       parameter=np.zeros_like(parameter), 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          sample=opt_sample,
                                                          level=level)

        return pivots, pvalues, intervals

    def estimate_covariance(self):
        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        P_diff = P_Z - P_ZE
        beta_estim = (self.D.T.dot(P_diff).dot(self.Y)) / (self.D.T.dot(P_diff).dot(self.D))

        n = self.Z.shape[0]
        #X = np.vstack([self.Y-self.D*beta_estim, self.D])
        cov_estim = (self.Y-self.D*beta_estim).dot(np.identity(n)-P_Z).dot(self.Y-self.D*beta_estim) / n

        return cov_estim


    @staticmethod
    def bigaussian_instance(n=1000,p=10,
                            s=3,snr=7.,random_signs=False, #true alpha parameter
                            gsnr_invalid = 1., #true gamma parameter
                            gsnr_valid = 1.,
                            beta = 1., #true beta parameter
                            Sigma = np.array([[1., 0.8], [0.8, 1.]]), #noise variance matrix
                            rho=0,scale=False,center=True): #Z matrix structure, note that scale=TRUE will simulate weak IV case!

        # Generate parameters
        # --> Alpha coefficients
        alpha = np.zeros(p) 
        alpha[:s] = snr 
        if random_signs:
            alpha[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
        active = np.zeros(p, np.bool)
        active[:s] = True
        # --> gamma coefficient
        #gamma = np.repeat([gsnr],p)
        gamma = np.ones(p)
        gamma[:s] *= gsnr_invalid
        gamma[s:] *= gsnr_valid

        # Generate samples
        # Generate Z matrix 
        Z = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
            np.sqrt(rho) * np.random.standard_normal(n)[:,None])
        if center:
            Z -= Z.mean(0)[None,:]
        if scale:
            Z /= (Z.std(0)[None,:] * np.sqrt(n))
        #    Z /= np.sqrt(n)
        # Generate error term
        mean = [0, 0]
        errorTerm = np.random.multivariate_normal(mean,Sigma,n)
        # Generate D and Y
        D = Z.dot(gamma) + errorTerm[:,1]
        Y = Z.dot(alpha) + D * beta + errorTerm[:,0]
    
        return Z, D, Y, alpha, beta, gamma



class lasso_iv_ar(highdim):

    r"""
    A class for the LASSO with invalid instrumental variables for post-selection inference.
    Specifically sampling for the Anderson-Rubin test statistic, note the sampling structure
    is different from the standard LASSO code.
    The problem solved is

    .. math::

        \text{minimize}_{\alpha, \beta} \frac{1}{2} \|P_Z (y-Z\alpha-D\beta)\|^2_2 + 
            \lambda \|\alpha\|_1 - \omega^T(\alpha \beta) + \frac{\epsilon}{2} \|(\alpha \beta)\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty.

    """

    # add .Z field for lasso IV subclass
    def __init__(self,
                 Y, 
                 D,
                 Z, 
                 penalty=None, 
                 ridge_term=None,
                 randomizer_scale=None):
    
        # form the projected design and response
        P_Z = Z.dot(np.linalg.pinv(Z))
        X = np.hstack([Z, D.reshape((-1,1))])
        P_ZX = P_Z.dot(X)
        P_ZY = P_Z.dot(Y)
        loglike = rr.glm.gaussian(P_ZX, P_ZY)

        n, p = Z.shape

        if penalty is None:
            penalty = 2.01 * np.sqrt(n * np.log(n))
        penalty = np.ones(loglike.shape[0]) * penalty
        penalty[-1] = 0.

        mean_diag = np.mean((P_ZX**2).sum(0))
        if ridge_term is None:
            #ridge_term = 1. * np.sqrt(n)
            ridge_term = (np.std(P_ZY) * np.sqrt(mean_diag) / np.sqrt(n - 1.))

        if randomizer_scale is None:
            #randomizer_scale = 0.5*np.sqrt(n)
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(P_ZY) * np.sqrt(n / (n - 1.))

        highdim.__init__(self, loglike, penalty, ridge_term, randomizer_scale)
        self.Z = Z
        self.D = D
        self.Y = Y


    # the ONLY change is .sampler line
    def fit(self, 
            solve_args={'tol':1.e-12, 'min_its':50}, 
            perturb=None):
        """
        Fit the randomized lasso using `regreg`.
        Parameters
        ----------
        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.
        Returns
        -------
        signs : np.float
             Support and non-zero signs of randomized lasso solution.
             
        """

        p = self.nfeature

        if perturb is None:
            perturb = self.randomizer.sample()
        self._initial_omega = perturb
        quad = rr.identity_quadratic(self.ridge_term, 0, -perturb)
        problem = rr.simple_problem(self.loglike, self.penalty)
        self.initial_soln = problem.solve(quad, **solve_args)

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        self.selection_variable = {'sign':_active_signs,
                                   'variables':self._overall}

        # initial state for opt variables

        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') + 
                            quad.objective(self.initial_soln, 'grad')) 
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.fabs(self.initial_soln[active])
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized])

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # observed state for score in internal coordinates

        self.observed_internal_state = np.hstack([_beta_unpenalized,
                                                  -self.loglike.smooth_objective(beta_bar, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self.num_opt_var))
        _score_linear_term = np.zeros((p, self.num_opt_var))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        _hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _hessian_unpen = np.dot(X.T, X[:, unpenalized] * W[:, None])

        _score_linear_term = -np.hstack([_hessian_active, _hessian_unpen])

        # set the observed score (data dependent) state

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p, j, active_signs[j]) for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = _hessian_active * active_signs[None, active] + self.ridge_term * active_directions
        _opt_linear_term[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), self.num_opt_var)
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (_hessian_unpen
                                                      + self.ridge_term * unpenalized_directions) 

        # two transforms that encode score and optimization
        # variable roles 

        self.opt_transform = (_opt_linear_term, self.initial_subgrad)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self._setup = True
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.ndim = self.loglike.shape[0]

        # compute implied mean and covariance

        cov, prec = self.randomizer.cov_prec
        opt_linear, opt_offset = self.opt_transform

        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        def log_density(logdens_linear, offset, cond_prec, score, opt):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)
        log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

        # now make the constraints

        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        affine_con = constraints(A_scaling,
                                 b_scaling,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        logdens_transform = (logdens_linear, opt_offset)

        self.sampler = affine_gaussian_sampler_iv(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               logdens_transform,
                                               selection_info=self.selection_variable) # should be signs and the subgradients we've conditioned on
        
        return active_signs


    def summary(self,
                parameter=None,
                Sigma_11=1.,
                level=0.95,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=False):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter beta_star -- defaults to 0.

        Sigma_11 : true Sigma_11, known for now

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = np.zeros(1)

        parameter = np.atleast_1d(parameter)

        # compute the observed_target for AR statistic
        # target = Z^T (I - P_{Z_E}) (Y-D \beta_0)

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))

        n, _ = self.Z.shape
        R_ZE = np.identity(n) - P_ZE

        self.K1 = self.Z.T.dot(R_ZE).dot(self.Y)
        self.K2 = self.Z.T.dot(R_ZE).dot(self.D)
        #observed_target = self.Z.T.dot(R_ZE).dot(self.Y - self.D * parameter)
        observed_target = self.K1 - self.K2 * parameter

        # only has the parametric version right now
        # compute cov_target, cov_target_score

        cov_target = self.Z.T.dot(R_ZE).dot(self.Z) * Sigma_11
        cov_target_score = np.hstack([self.Z, self.D.reshape((-1,1))]).T.dot(R_ZE).dot(self.Z)
        cov_target_score *= Sigma_11

        # this is for Anderson-Rubin
        alternatives = ['greater']

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target,
                                                  cov_target,
                                                  cov_target_score,
                                                  self.K2,
                                                  self.test_stat,
                                                  parameter=parameter, 
                                                  sample=opt_sample,
                                                  alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       self.K2,
                                                       self.test_stat,
                                                       parameter=np.zeros_like(parameter), 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          self.K2,
                                                          self.test_stat,
                                                          parameter=parameter,
                                                          sample=opt_sample,
                                                          level=level,
                                                          how_many_sd=40)

        return pivots, pvalues, intervals

    def test_stat(self, parameter, target):
        # target is of n by p
        target = np.atleast_2d(target)

        n, p = self.Z.shape
        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        R_ZE = np.identity(n) - P_ZE
        denom = (self.Y-self.D*parameter).dot(R_ZE).dot(self.Y-self.D*parameter) / (n-p)
        ZTZ_inv = np.linalg.inv(self.Z.T.dot(self.Z))
        result = np.sum(target * (ZTZ_inv.dot(target.T).T), axis=1)

        return result


    def estimate_covariance(self):
        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        P_diff = P_Z - P_ZE
        beta_estim = (self.D.T.dot(P_diff).dot(self.Y)) / (self.D.T.dot(P_diff).dot(self.D))

        n = self.Z.shape[0]
        #X = np.vstack([self.Y-self.D*beta_estim, self.D])
        cov_estim = (self.Y-self.D*beta_estim).dot(np.identity(n)-P_Z).dot(self.Y-self.D*beta_estim) / n

        return cov_estim


    @staticmethod
    def bigaussian_instance(n=1000,p=10,
                            s=3,snr=7.,random_signs=False, #true alpha parameter
                            gsnr_invalid = 1., #true gamma parameter
                            gsnr_valid = 1.,
                            beta = 1., #true beta parameter
                            Sigma = np.array([[1., 0.8], [0.8, 1.]]), #noise variance matrix
                            rho=0,scale=False,center=True): #Z matrix structure, note that scale=TRUE will simulate weak IV case!

        # Generate parameters
        # --> Alpha coefficients
        alpha = np.zeros(p) 
        alpha[:s] = snr 
        if random_signs:
            alpha[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
        active = np.zeros(p, np.bool)
        active[:s] = True
        # --> gamma coefficient
        #gamma = np.repeat([gsnr],p)
        gamma = np.ones(p)
        gamma[:s] *= gsnr_invalid
        gamma[s:] *= gsnr_valid

        # Generate samples
        # Generate Z matrix 
        Z = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
            np.sqrt(rho) * np.random.standard_normal(n)[:,None])
        if center:
            Z -= Z.mean(0)[None,:]
        if scale:
            Z /= (Z.std(0)[None,:] * np.sqrt(n))
        #    Z /= np.sqrt(n)
        # Generate error term
        mean = [0, 0]
        errorTerm = np.random.multivariate_normal(mean,Sigma,n)
        # Generate D and Y
        D = Z.dot(gamma) + errorTerm[:,1]
        Y = Z.dot(alpha) + D * beta + errorTerm[:,0]
    
        return Z, D, Y, alpha, beta, gamma



class affine_gaussian_sampler_iv(affine_gaussian_sampler):

    def confidence_intervals(self,
                             observed_target,
                             target_cov,
                             score_cov,
                             linear_func,
                             test_stat,
                             beta_reference,
                             sample_args=(),
                             sample=None,
                             level=0.9,
                             how_many_sd=20):
        '''

        Parameters
        ----------

        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.

        sample_args : sequence
           Arguments to `self.sample` if sample is None.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.

        level : float (optional)
            Specify the
            confidence level.

        Notes
        -----

        Construct selective confidence intervals
        for each parameter of the target.

        Returns
        -------

        intervals : [(float, float)]
            List of confidence intervals.
        '''

        if sample is None:
            sample = self.sample(*sample_args)
        else:
            ndraw = sample.shape[0]

        _intervals = optimization_intervals_iv([(self, sample, target_cov, score_cov)],
                                            observed_target, ndraw)

        limits = []

        #for i in range(observed_target.shape[0]):
        #    keep = np.zeros_like(observed_target)
        #    keep[i] = 1.
        #    limits.append(_intervals.confidence_interval(keep, level=level))
        limits.append(_intervals.confidence_interval(linear_func, test_stat, beta_reference, level=level, how_many_sd=how_many_sd))

        return np.array(limits)

    def coefficient_pvalues(self,
                            observed_target,
                            target_cov,
                            score_cov,
                            linear_func,
                            test_stat,
                            parameter=None,
                            sample_args=(),
                            sample=None,
                            alternatives=None):
        '''
        Construct selective p-values
        for each parameter of the target.

        Parameters
        ----------

        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.

        parameter : np.float (optional)
            A vector of parameters with shape `self.shape`
            at which to evaluate p-values. Defaults
            to `np.zeros(self.shape)`.

        sample_args : sequence
           Arguments to `self.sample` if sample is None.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.

        alternatives : list of ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------
        pvalues : np.float

        '''

        if alternatives is None:
            #alternatives = ['twosided'] * observed_target.shape[0]
            alternatives = ['greater']

        if sample is None:
            sample = self.sample(*sample_args)
        else:
            ndraw = sample.shape[0]

        if parameter is None:
            #parameter = np.zeros(observed_target.shape[0])
            parameter = np.zeros(1)

        _intervals = optimization_intervals_iv([(self, sample, target_cov, score_cov)],
                                            observed_target, ndraw)
        pvals = []

        #for i in range(observed_target.shape[0]):
        #    keep = np.zeros_like(observed_target)
        #    keep[i] = 1.
        #    pvals.append(_intervals.pivot(keep, candidate=parameter[i], alternative=alternatives[i]))
        pvals.append(_intervals.pivot(linear_func, parameter, 0., test_stat, 'greater'))

        return np.array(pvals)


class optimization_intervals_iv(optimization_intervals):

    def pivot(self,
              linear_func,
              parameter_reference, 
              candidate,
              test_stat,
              alternative='twosided'):
        '''
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalue : np.float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        # observed target at beta_reference
        observed_target = self.observed
        sample_target = self._normal_sample

        target_cov = self.target_cov

        nuisance = []
        translate_dirs = []
        translate_dirs_candidate = []
        for opt_sampler, opt_sample, _, score_cov in self.opt_sampling_info:
            #cur_score_cov = linear_func.dot(score_cov)
            cur_score_cov = score_cov

            # cur_nuisance is in the view's score coordinates
            cur_nuisance = opt_sampler.observed_score_state - cur_score_cov.dot(np.linalg.inv(target_cov)).dot(observed_target)
            nuisance.append(cur_nuisance)
            translate_dirs.append(cur_score_cov.dot(np.linalg.inv(target_cov)))
            translate_dirs_candidate.append(-cur_score_cov.dot(np.linalg.inv(target_cov)).dot(linear_func))


        weights = self._weights(sample_target,  # normal sample under zero
                                candidate, # the difference beta-beta_reference
                                nuisance,                 # nuisance sufficient stats for each view
                                translate_dirs,           # points will be moved like sample * score_cov
                                translate_dirs_candidate)               
        
        
        #pivot = np.mean((sample_stat + candidate <= observed_stat) * weights) / np.mean(weights)
        beta = parameter_reference + candidate
        observed_target_beta = observed_target - linear_func*candidate
        observed_stat = test_stat(beta, observed_target)
        sample_stat = test_stat(beta, sample_target)
        pivot = np.mean((sample_stat <= observed_stat) * weights) / np.mean(weights)

        if alternative == 'twosided':
            return 2 * min(pivot, 1 - pivot)
        elif alternative == 'less':
            return pivot
        else:
            return 1 - pivot

    def confidence_interval(self, linear_func, test_stat, beta_reference, level=0.90, how_many_sd=20):

        sample_stat = test_stat(beta_reference, self._normal_sample)
        #observed_stat = test_stat(self.observed.dot(linear_func))
        
        #_norm = np.linalg.norm(linear_func)
        grid_min, grid_max = -how_many_sd * np.std(sample_stat), how_many_sd * np.std(sample_stat)

        #def _rootU(gamma):
            #return self.pivot(linear_func,
            #                  beta_reference,
            #                  gamma,
            #                  test_stat,
            #                  alternative='less') - (1 - level) / 2.
        #def _rootL(gamma):
            #return self.pivot(linear_func,
            #                  beta_reference,
            #                  gamma,
            #                  test_stat,
            #                  alternative='less') - (1 + level) / 2.

        #upper = bisect(_rootU, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))
        #lower = bisect(_rootL, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))

        def _root(gamma):
            return self.pivot(linear_func,
                              beta_reference,
                              gamma,
                              test_stat,
                              alternative='greater') - (1 - level)

        upper = bisect(_root, 0., grid_max, xtol=1.e-5*(grid_max - grid_min))
        lower = bisect(_root, grid_min, 0., xtol=1.e-5*(grid_max - grid_min))

        return lower + beta_reference, upper + beta_reference

    # Private methods

    def _weights(self, 
                 sample_target,
                 candidate,
                 nuisance,
                 translate_dirs,
                 translate_dirs_candidate):

        # Here we should loop through the views
        # and move the score of each view 
        # for each projected (through linear_func) normal sample
        # using the linear decomposition

        # We need access to the map that takes observed_score for each view
        # and constructs the full randomization -- this is the reconstruction map
        # for each view

        # The data state for each view will be set to be N_i + A_i \hat{\theta}_i
        # where N_i is the nuisance sufficient stat for the i-th view's
        # data with respect to \hat{\theta} and N_i  will not change because
        # it depends on the observed \hat{\theta} and observed score of i-th view

        # In this function, \hat{\theta}_i will change with the Monte Carlo sample

        score_sample = []
        _lognum = 0
        for i, opt_info in enumerate(self.opt_sampling_info):
            opt_sampler, opt_sample = opt_info[:2]
            score_sample = translate_dirs[i].dot(sample_target.T).T + nuisance[i][None, :] # these are now score coordinates
            score_sample += (translate_dirs_candidate[i]*candidate)[None, :]
            _lognum += opt_sampler.log_density(score_sample, opt_sample)

        _logratio = _lognum - self._logden
        _logratio -= _logratio.max()

        return np.exp(_logratio)



# this method uses highdim class with 'selected' target which theoretically should give the desired result
# simulation result works and it seems the way to set ridge_term and randomizer_scale matters       
def lasso_iv_selected(gsnr=1., beta=1., sigma_12=0.8, ndraw=5000, burnin=1000):

    Z, D, Y, alpha, beta, gamma = lasso_iv.bigaussian_instance(gsnr=gsnr, beta=beta, Sigma = np.array([[1., sigma_12], [sigma_12, 1.]]))
    PZ = Z.dot(np.linalg.pinv(Z))

    n, p = Z.shape

    penalty = 2.01 * np.sqrt(n * np.log(n))
    penalty = np.ones(p + 1) * penalty
    penalty[-1] = 0.

    L = highdim.gaussian(PZ.dot(np.hstack([Z, D.reshape((-1,1))])), PZ.dot(Y), penalty)
    signs = L.fit()
    nonzero = np.nonzero(signs != 0)[0]

    if p not in set(nonzero):
        raise ValueError('last should always be selected!')

    if set(nonzero).issuperset(np.nonzero(alpha)[0]) and len(nonzero) <= p :
        parameter = np.hstack([alpha[nonzero[:-1]], beta])

        pivots, pval, intervals = L.summary(target="selected",
                                            parameter=parameter,
                                            ndraw=ndraw,
                                            burnin=burnin,
                                            level=0.95, 
                                            compute_intervals=True,
                                            dispersion=1.)
        return pivots[-1:], (intervals[-1][0] < beta) * (intervals[-1][1] > beta)

    return [], np.nan


# use the tsls statistic instead of tsls beta as target
# the null pivot is usable but CI is WRONG because of the importance weight
class stat_lasso_iv(lasso_iv):

    def summary(self,
                parameter=None,
                Sigma_11=1.,
                level=0.95,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=False):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter beta_star -- defaults to 0.

        Sigma_11 : true Sigma_11, known for now

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = 0.

        # compute tsls, i.e. the observed_target

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        P_ZX, P_ZY = self.loglike.data
        P_ZD = P_ZX[:,-1]
        two_stage_ls = (P_ZD.dot(P_Z-P_ZE).dot(P_ZY-P_ZD*parameter))/np.sqrt(Sigma_11*P_ZD.dot(P_Z-P_ZE).dot(P_ZD))
        #denom = P_ZD.dot(P_Z - P_ZE).dot(P_ZD)
        #two_stage_ls = (P_ZD.dot(P_Z - P_ZE).dot(P_ZY)) / denom
        two_stage_ls = np.atleast_1d(two_stage_ls)
        observed_target = two_stage_ls

        # only has the parametric version right now
        # compute cov_target, cov_target_score

        cov_target = np.atleast_2d(1.)

        cov_target_score = -1.*np.sqrt(Sigma_11/P_ZD.dot(P_Z-P_ZE).dot(P_ZD))*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        #cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        cov_target_score = np.atleast_2d(cov_target_score)

        alternatives = ['twosided']

        opt_sample = self.sampler.sample(ndraw, burnin)

        # set parameter to be zero!
        pivots = self.sampler.coefficient_pvalues(observed_target, 
                                                  cov_target, 
                                                  cov_target_score, 
                                                  sample=opt_sample,
                                                  alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          sample=opt_sample)

        return pivots, pvalues, intervals






