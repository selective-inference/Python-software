import functools # for bootstrap partial mapping

import numpy as np
from scipy.stats import norm as ndist

from regreg.api import glm, identity_quadratic

from .base import restricted_estimator

import regreg.api as rr
import regreg.affine as ra

def pairs_bootstrap_glm(glm_loss,
                        active, 
                        beta_full=None, 
                        inactive=None, 
                        scaling=1.,
                        solve_args={'min_its':50, 'tol':1.e-10}):
    """
    Construct a non-parametric bootstrap sampler that 
    samples the estimates ($\bar{\beta}_E^*$) of a generalized 
    linear model (GLM) restricted to `active`
    as well as, optionally, the inactive coordinates of the score of the 
    GLM evaluated at the estimates ($\nabla \ell(\bar{\beta}_E)[-E]$) where
    $\bar{\beta}_E$ is padded with zeros where necessary.
    
    Parameters
    ----------

    glm_loss : regreg.smooth.glm.glm
        The loss of the generalized linear model.

    active : np.bool
        Boolean indexing array

    beta_full : np.float (optional)
        Solution to the restricted problem, zero except where active is nonzero.

    inactive : np.bool (optional)
        Boolean indexing array

    scaling : float
        Scaling to keep entries of roughly constant order. Active entries
        are multiplied by sqrt(scaling) inactive ones are divided
        by sqrt(scaling).

    solve_args : dict
        Arguments passed to solver of restricted problem (`restricted_estimator`) if 
        beta_full is None.

    Returns
    -------

    bootstrap_sampler : callable
        A callable object that takes a sample of indices and returns
        the corresponding bootstrap sample.

    """
    X, Y = glm_loss.data

    if beta_full is None:
        beta_active = restricted_estimator(glm_loss, active, solve_args=solve_args)
        beta_full = np.zeros(glm_loss.shape)
        beta_full[active] = beta_active
    else:
        beta_active = beta_full[active]

    X_active = X[:,active]

    nactive = active.sum()
    ntotal = nactive

    if inactive is not None:
        X_inactive = X[:,inactive]
        ntotal += inactive.sum()

    _bootW = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))
    _bootQ = X_active.T.dot(_bootW.dot(X_active))
    _bootQinv = np.linalg.inv(_bootQ)
    if inactive is not None:
        _bootC = X_inactive.T.dot(_bootW.dot(X_active))
        _bootI = _bootC.dot(_bootQinv)
    else:
        _bootI = None

    nactive = active.sum()
    if inactive is not None:
        X_full = np.hstack([X_active, X_inactive])
        beta_overall = np.zeros(X_full.shape[1])
        beta_overall[:nactive] = beta_active
    else:
        X_full = X_active
        beta_overall = beta_active

    _boot_mu = lambda X_full, beta_overall: glm_loss.saturated_loss.mean_function(X_full.dot(beta_overall))

    if ntotal > nactive:
        observed = np.hstack([beta_active, -glm_loss.smooth_objective(beta_full, 'grad')[inactive]])
    else:
        observed = beta_active

    # scaling is a lipschitz constant for a gradient squared
    _sqrt_scaling = np.sqrt(scaling)

    def _boot_score(X_full, Y, ntotal, _bootQinv, _bootI, nactive, _sqrt_scaling, beta_overall, indices):
        X_star = X_full[indices]
        Y_star = Y[indices]
        score = X_star.T.dot(Y_star - _boot_mu(X_star, beta_overall))
        result = np.zeros(ntotal)
        result[:nactive] = _bootQinv.dot(score[:nactive])
        if ntotal > nactive:
            result[nactive:] = score[nactive:] - _bootI.dot(score[:nactive])
        result[:nactive] *= _sqrt_scaling
        result[nactive:] /= _sqrt_scaling
        return result

    observed[:nactive] *= _sqrt_scaling
    observed[nactive:] /= _sqrt_scaling

    return functools.partial(_boot_score, X_full, Y, ntotal, _bootQinv, _bootI, nactive, _sqrt_scaling, beta_overall), observed

def pairs_inactive_score_glm(glm_loss, 
                             active, 
                             beta_active, 
                             scaling=1.,
                             inactive=None,
                             solve_args={'min_its':50, 'tol':1.e-10}):

    """
    Construct a non-parametric bootstrap sampler that 
    samples the inactive coordinates of the score of the 
    GLM evaluated at the estimates ($\nabla \ell(\bar{\beta}_E)[-E]$) where
    $\bar{\beta}_E$ is padded with zeros where necessary.
    
    Parameters
    ----------

    glm_loss : regreg.smooth.glm.glm
        The loss of the generalized linear model.

    active : np.bool
        Boolean indexing array

    beta_active : np.float (optional)
        Solution to the restricted problem.

    scaling : float
        Scaling to keep entries of roughly constant order. Active entries
        are multiplied by sqrt(scaling) inactive ones are divided
        by sqrt(scaling).

    inactive : np.bool (optional)
        Which coordinates to return. If None, defaults
        to ~active.

    solve_args : dict
        Arguments passed to solver of restricted problem (`restricted_estimator`) if 
        beta_full is None.

    Returns
    -------

    bootstrap_sampler : callable
        A callable object that takes a sample of indices and returns
        the corresponding bootstrap sample.

    """

    if inactive is None:
        inactive = ~active

    beta_full = np.zeros(glm_loss.shape)
    beta_full[active] = beta_active

    _full_boot_score = pairs_bootstrap_glm(glm_loss, 
                                           active, 
                                           beta_full=beta_full,
                                           inactive=inactive,
                                           scaling=scaling,
                                           solve_args=solve_args)[0]
    nactive = active.sum()

    def _boot_score(indices):
        return _full_boot_score(indices)[nactive:]

    return _boot_score


def pairs_bootstrap_score(glm_loss,
                          active, 
                          beta_active=None, 
                          solve_args={'min_its':50, 'tol':1.e-10}):
    """
    Construct a non-parametric bootstrap sampler that 
    samples the score ($\nabla \ell(\bar{\beta}_E)) ofa generalized 
    linear model (GLM) restricted to `active`
    as well as, optionally, the inactive coordinates of the score of the 
    GLM evaluated at the estimates ($\nabla \ell(\bar{\beta}_E)[-E]$) where
    $\bar{\beta}_E$ is padded with zeros where necessary.
    
    Parameters
    ----------

    glm_loss : regreg.smooth.glm.glm
        The loss of the generalized linear model.

    active : np.bool
        Boolean indexing array

    beta_active : np.float (optional)
        Solution to the restricted problem. 

    solve_args : dict
        Arguments passed to solver of restricted problem (`restricted_estimator`) if 
        beta_full is None.

    Returns
    -------

    bootstrap_sampler : callable
        A callable object that takes a sample of indices and returns
        the corresponding bootstrap sample.

    """

    X, Y = glm_loss.data

    if beta_active is None:
        beta_active = restricted_estimator(glm_loss, active, solve_args=solve_args)
    X_active = X[:,active]

    _bootW = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))

    _boot_mu = lambda X_active, beta_active: glm_loss.saturated_loss.mean_function(X_active.dot(beta_active))

    def _boot_score(X, Y, active, beta_active, indices):
        X_star = X[indices]
        Y_star = Y[indices]
        score = -X_star.T.dot(Y_star - _boot_mu(X_star[:,active], beta_active))
        return score

    return functools.partial(_boot_score, X, Y, active, beta_active)

def set_alpha_matrix(glm_loss,
                     active,
                     beta_full=None,
                     inactive=None,
                     scaling=1.,
                     solve_args={'min_its': 50, 'tol': 1.e-10}):
    """
    DESCRIBE WHAT THIS DOES

    Parameters
    ----------

    glm_loss : regreg.smooth.glm.glm
        The loss of the generalized linear model.

    active : np.bool
        Boolean indexing array

    beta_full : np.float (optional)
        Solution to the restricted problem, zero except where active is nonzero.

    inactive : np.bool (optional)
        Boolean indexing array

    scaling : float
        Scaling to keep entries of roughly constant order. Active entries
        are multiplied by sqrt(scaling) inactive ones are divided
        by sqrt(scaling).

    solve_args : dict
        Arguments passed to solver of restricted problem (`restricted_estimator`) if 
        beta_full is None.

    Returns
    -------

    ???????

    """
    X, Y = glm_loss.data

    if beta_full is None:
        beta_active = restricted_estimator(glm_loss, active, solve_args=solve_args)
        beta_full = np.zeros(glm_loss.shape)
        beta_full[active] = beta_active
    else:
        beta_active = beta_full[active]

    X_active = X[:,active]

    nactive = active.sum()
    ntotal = nactive

    if inactive is not None:
        X_inactive = X[:,inactive]
        ntotal += inactive.sum()

    _W = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))
    _Q = X_active.T.dot(_W.dot(X_active))
    _Qinv = np.linalg.inv(_Q)
    nactive = active.sum()
    if inactive is not None:
        X_full = np.hstack([X_active, X_inactive])
        beta_overall = np.zeros(X_full.shape[1])
        beta_overall[:nactive] = beta_active
    else:
        X_full = X_active
        beta_overall = beta_active

    obs_residuals = Y - glm_loss.saturated_loss.mean_function(X_full.dot(beta_overall))

    return np.dot(np.dot(_Qinv, X_active.T), np.diag(obs_residuals))

# Methods to form appropriate covariances

def bootstrap_cov(sampler, boot_target, cross_terms=(), nsample=2000):
    """
    m out of n bootstrap

    returns estimates of covariance matrices: boot_target with itself,
    and the blocks of (boot_target, boot_other) for other in cross_terms

    """

    _mean_target = 0.
    if len(cross_terms) > 0:
        _mean_cross = [0.] * len(cross_terms)
        _outer_cross = [0.] * len(cross_terms)
    _outer_target = 0.

    for j in range(nsample):
        indices = sampler()
        _boot_target = boot_target(indices)

        _mean_target += _boot_target
        _outer_target += np.multiply.outer(_boot_target, _boot_target)

        for i, _boot in enumerate(cross_terms):
            _boot_sample = _boot(indices)
            _mean_cross[i] += _boot_sample
            _outer_cross[i] += np.multiply.outer(_boot_target, _boot_sample)

    _mean_target /= nsample
    _outer_target /= nsample

    for i in range(len(cross_terms)):
        _mean_cross[i] /= nsample
        _outer_cross[i] /= nsample

    _cov_target = _outer_target - np.multiply.outer(_mean_target, _mean_target)
    if len(cross_terms) == 0:
        return _cov_target
    return [_cov_target] + [_o - np.multiply.outer(_mean_target, _m) for _m, _o in zip(_mean_cross, _outer_cross)]

def glm_nonparametric_bootstrap(m, n):
    """
    The m out of n bootstrap.
    """
    return functools.partial(bootstrap_cov, lambda: np.random.choice(n, size=(m,), replace=True))

def resid_bootstrap(gaussian_loss,
                    active, # boolean
                    inactive=None,
                    scaling=1.):

    X, Y = gaussian_loss.data
    X_active = X[:,active]

    nactive = active.sum()
    ntotal = nactive

    if inactive is not None:
        X_inactive = X[:,inactive]
        ntotal += inactive.sum()

    X_active_inv = np.linalg.pinv(X_active)
    beta_active = X_active_inv.dot(Y)

    if ntotal > nactive:
        beta_full = np.zeros(X.shape[1])
        beta_full[active] = beta_active
        observed = np.hstack([beta_active, -gaussian_loss.smooth_objective(beta_full, 'grad')[inactive]])
    else:
        observed = beta_active

    if ntotal > nactive:
        X_inactive = X[:,inactive]
        X_inactive_resid = X_inactive - X_active.dot(X_active_inv.dot(X_inactive))

    _sqrt_scaling = np.sqrt(scaling)

    def _boot_score(Y_star):
        beta_hat = X_active_inv.dot(Y_star)
        result = np.zeros(ntotal)
        result[:nactive] = beta_hat
        if ntotal > nactive:
            result[nactive:] = X_inactive_resid.T.dot(Y_star)
        result[:nactive] *= _sqrt_scaling
        result[nactive:] /= _sqrt_scaling
        return result

    return _boot_score, observed

def parametric_cov(glm_loss, 
                   target_with_linear_func, 
                   cross_terms=(),
                   dispersion=None,
                   solve_args={'min_its':50, 'tol':1.e-10}):

    # cross_terms are different active sets

    target, linear_func = target_with_linear_func

    target_bool = np.zeros(glm_loss.shape, np.bool)
    target_bool[target] = True
    target = target_bool

    linear_funcT = linear_func.T

    X, Y = glm_loss.data
    n, p = X.shape

    def _WQ(active):
        beta_active = restricted_estimator(glm_loss, active, solve_args=solve_args)
        W = glm_loss.saturated_loss.hessian(X[:,active].dot(beta_active))
        return W

    # weights and Q at the target
    W_T = _WQ(target)

    X_T = X[:,target]
    XW_T = W_T[:, None] * X_T
    Q_T_inv = np.linalg.inv(X_T.T.dot(XW_T))

    beta_T = restricted_estimator(glm_loss, target, solve_args=solve_args)

    # this is Pearson's X^2 dispersion estimator
    if dispersion is None:
        sigma_T = np.sqrt(np.sum((Y-glm_loss.saturated_loss.mean_function(X_T.dot(beta_T)))**2 / W_T)/(n-np.sum(target)))
    else:
        sigma_T = dispersion

    covariances = [linear_func.dot(Q_T_inv).dot(linear_funcT) * (sigma_T**2)]

    for cross in cross_terms:
        # the covariances are for (\bar{\beta}_{C}, N_C) -- C for cross

        cross_bool = np.zeros(X.shape[1], np.bool)
        cross_bool[cross] = True; cross = cross_bool

        X_C = X[:, cross]
        X_IT = X[:, ~cross].T
        Q_C_inv = np.linalg.inv(X_C.T.dot(W_T[:, None] * X_C))
        beta_block = Q_C_inv.dot(X[:, cross].T.dot(XW_T)).dot(Q_T_inv)
        null_block = X_IT.dot(XW_T) - X_IT.dot(W_T[:, None] * X_C).dot(Q_C_inv).dot(X[:, cross].T.dot(XW_T))
        null_block = null_block.dot(Q_T_inv)

        beta_C = restricted_estimator(glm_loss, cross, solve_args=solve_args)
        if dispersion is None:
            sigma_C = sigma_T # Hmm... not sure here 
            # sigma_C = np.sqrt(np.sum((Y - glm_loss.saturated_loss.mean_function(X_C.dot(beta_C)) / W_C) ** 2) / (n - np.sum(cross)))
        else:
            sigma_C = dispersion

        covariances.append(np.vstack([beta_block, null_block]).dot(linear_funcT).T * sigma_T * sigma_C)

    return covariances


def glm_parametric_covariance(glm_loss, solve_args={'min_its':50, 'tol':1.e-10}):
    """
    A constructor for parametric covariance
    """
    return functools.partial(parametric_cov, glm_loss, solve_args=solve_args)


def standard_split_ci(glm_loss, X, y, active, leftout_indices, alpha=0.1):
    """
    Data plitting confidence intervals via bootstrap.
    """
    loss = glm_loss(X[leftout_indices,], y[leftout_indices])
    boot_target, target_observed = pairs_bootstrap_glm(loss, active)
    nactive = np.sum(active)
    size= np.sum(leftout_indices)
    observed = target_observed[:nactive]
    boot_target_restricted = lambda indices: boot_target(indices)[:nactive]
    sampler = lambda: np.random.choice(size, size=(size,), replace=True)
    target_cov = bootstrap_cov(sampler, boot_target_restricted)

    quantile = - ndist.ppf(alpha / float(2))
    LU = np.zeros((2, target_observed.shape[0]))
    for j in range(observed.shape[0]):
        sigma = np.sqrt(target_cov[j, j])
        LU[0, j] = observed[j] - sigma * quantile
        LU[1, j] = observed[j] + sigma * quantile
    return LU.T

