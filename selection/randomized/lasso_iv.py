"""
Classes encapsulating some common workflows in randomized setting
"""

from copy import copy
import functools

import numpy as np
import regreg.api as rr
from .lasso import highdim
#from .randomization import randomization
#from .query import multiple_queries, optimization_sampler
#from .M_estimator import restricted_Mest

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






# rescaled version of lasso_iv which uses the scaled \sqrt{n} beta_hat as target
# only need to change observed_target, cov_target, cov_target_score and also parameter input
class rescaled_lasso_iv(lasso_iv):

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

        n, _ = self.Z.shape
        parameter *= np.sqrt(n)
        parameter = np.atleast_1d(parameter)

        # compute tsls, i.e. the observed_target

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        P_ZX, P_ZY = self.loglike.data
        P_ZD = P_ZX[:,-1]
        #two_stage_ls = (P_ZD.dot(P_Z-P_ZE).dot(self.P_ZY-P_ZD*parameter))/np.sqrt(Sigma_11*P_ZD.dot(P_Z-P_ZE).dot(P_ZD))
        denom = P_ZD.dot(P_Z - P_ZE).dot(P_ZD)
        two_stage_ls = (P_ZD.dot(P_Z - P_ZE).dot(P_ZY)) / denom
        two_stage_ls = np.atleast_1d(two_stage_ls)
        observed_target = two_stage_ls
        observed_target *= np.sqrt(n)

        # only has the parametric version right now
        # compute cov_target, cov_target_score

        cov_target = np.atleast_2d(Sigma_11/denom)
        cov_target *= n
        #score_cov = -1.*np.sqrt(Sigma_11/P_ZD.dot(P_Z-P_ZE).dot(P_ZD))*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        cov_target_score = np.atleast_2d(cov_target_score)
        cov_target_score *= np.sqrt(n)

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
                                                          sample=opt_sample)

        return pivots, pvalues, intervals


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






