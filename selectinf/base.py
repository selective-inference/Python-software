import numpy as np

import regreg.api as rr
import regreg.affine as ra

from .algorithms.debiased_lasso import (debiasing_matrix,
                                        pseudoinverse_debiasing_matrix)

def restricted_estimator(loss, active, solve_args={'min_its':50, 'tol':1.e-10}):
    """
    Fit a restricted model using only columns `active`.

    Parameters
    ----------

    Mest_loss : objective function
        A GLM loss.

    active : ndarray
        Which columns to use.

    solve_args : dict
        Passed to `solve`.

    Returns
    -------

    soln : ndarray
        Solution to restricted problem.

    """
    X, Y = loss.data

    if not loss._is_transform and hasattr(loss, 'saturated_loss'): # M_est is a glm
        X_restricted = X[:,active]
        loss_restricted = rr.affine_smooth(loss.saturated_loss, X_restricted)
    else:
        I_restricted = ra.selector(active, ra.astransform(X).input_shape[0], ra.identity((active.sum(),)))
        loss_restricted = rr.affine_smooth(loss, I_restricted.T)
    beta_E = loss_restricted.solve(**solve_args)
    
    return beta_E


# functions construct targets of inference
# and covariance with score representation

def selected_targets(loglike, 
                     solution,
                     features=None,
                     sign_info={}, 
                     dispersion=None,
                     solve_args={'tol': 1.e-12, 'min_its': 100},
                     hessian=None):

    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape

    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    linpred = X[:, features].dot(observed_target)
    
    Hfeat = _compute_hessian(loglike,
                             solution,
                             features)[1]
    Qfeat = Hfeat[features]
    _score_linear = -Hfeat

    cov_target = np.linalg.inv(Qfeat)
    crosscov_target_score = _score_linear.dot(cov_target)
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                observed_target.shape[0])

    regress_target_score = np.zeros((cov_target.shape[0], p))
    regress_target_score[:,features] = cov_target
    return observed_target, cov_target * dispersion, regress_target_score, dispersion, alternatives

def full_targets(loglike, 
                 solution,
                 features=None,
                 dispersion=None,
                 solve_args={'tol': 1.e-12, 'min_its': 50},
                 hessian=None):
    
    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape
    features_bool = np.zeros(p, np.bool)
    features_bool[features] = True
    features = features_bool

    # target is one-step estimator

    full_estimator = loglike.solve(**solve_args)
    linpred = X.dot(full_estimator)
    Qfull = _compute_hessian(loglike,
                             full_estimator)

    Qfull_inv = np.linalg.inv(Qfull)
    cov_target = Qfull_inv[features][:, features]
    observed_target = full_estimator[features]
    crosscov_target_score = np.zeros((p, cov_target.shape[0]))
    crosscov_target_score[features] = -np.identity(cov_target.shape[0])

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                p)

    alternatives = ['twosided'] * features.sum()
    regress_target_score = Qfull_inv[features] # weights missing?
    return observed_target, cov_target * dispersion, regress_target_score, dispersion, alternatives

def debiased_targets(loglike, 
                     solution,
                     features=None,
                     sign_info={}, 
                     penalty=None, #required kwarg
                     dispersion=None,
                     approximate_inverse='JM',
                     debiasing_args={}):

    if features is None:
        features = solution != 0

    if penalty is None:
        raise ValueError('require penalty for consistent estimator')

    X, y = loglike.data
    n, p = X.shape
    features_bool = np.zeros(p, np.bool)
    features_bool[features] = True
    features = features_bool

    # relevant rows of approximate inverse

    linpred = X.dot(solution)
    W = loglike.saturated_loss.hessian(linpred)
    if approximate_inverse == 'JM':
        Qinv_hat = np.atleast_2d(debiasing_matrix(X * np.sqrt(W)[:, None], 
                                                  np.nonzero(features)[0],
                                                  **debiasing_args)) / n
    else:
        Qinv_hat = np.atleast_2d(pseudoinverse_debiasing_matrix(X * np.sqrt(W)[:, None],
                                                                np.nonzero(features)[0],
                                                                **debiasing_args))

    problem = rr.simple_problem(loglike, penalty)
    nonrand_soln = problem.solve()
    G_nonrand = loglike.smooth_objective(nonrand_soln, 'grad')

    observed_target = nonrand_soln[features] - Qinv_hat.dot(G_nonrand)

    Qfull, Qrelax = _compute_hessian(loglike,
                                     solution,
                                     features)

    if p > n:
        M1 = Qinv_hat.dot(X.T)
        cov_target = (M1 * W[None, :]).dot(M1.T)
        crosscov_target_score = -(M1 * W[None, :]).dot(X).T
    else:
        Qfull = X.T.dot(W[:, None] * X)
        cov_target = Qinv_hat.dot(Qfull.dot(Qinv_hat.T))
        crosscov_target_score = -Qinv_hat.dot(Qfull).T

    if dispersion is None:  # use Pearson's X^2
        relaxed_soln = nonrand_soln[features] - np.linalg.inv(Qrelax[features]).dot(G_nonrand[features])
        Xfeat = X[:, features]
        linpred =  Xfeat.dot(relaxed_soln)
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                features.sum())

    alternatives = ['twosided'] * features.sum()
    return observed_target, cov_target * dispersion, Qinv_hat, dispersion, alternatives

def form_targets(target, 
                 loglike, 
                 solution,
                 features, 
                 **kwargs):
    _target = {'full':full_targets,
               'selected':selected_targets,
               'debiased':debiased_targets}[target]
    return _target(loglike,
                   solution,
                   features,
                   **kwargs)

def _compute_hessian(loglike,
                     beta_bar,
                     *bool_indices):

    X, y = loglike.data
    linpred = X.dot(beta_bar)
    n = linpred.shape[0]

    if hasattr(loglike.saturated_loss, "hessian"): # a GLM -- all we need is W
        W = loglike.saturated_loss.hessian(linpred)
        parts = [np.dot(X.T, X[:, bool_idx] * W[:, None]) for bool_idx in bool_indices]
        _hessian = np.dot(X.T, X * W[:, None]) # CAREFUL -- this will be big
    elif hasattr(loglike.saturated_loss, "hessian_mult"):
        parts = []
        for bool_idx in bool_indices:
            _right = np.zeros((n, bool_idx.sum()))
            for i, j in enumerate(np.nonzero(bool_idx)[0]):
                _right[:,i] = loglike.saturated_loss.hessian_mult(linpred, 
                                                                       X[:,j], 
                                                                       case_weights=loglike.saturated_loss.case_weights)
            parts.append(X.T.dot(_right))
        _hessian = np.zeros_like(X)
        for i in range(X.shape[1]):
            _hessian[:,i] = loglike.saturated_loss.hessian_mult(linpred, 
                                                                     X[:,i], 
                                                                     case_weights=loglike.saturated_loss.case_weights)
        _hessian = X.T.dot(_hessian)
    else:
        raise ValueError('saturated_loss has no hessian or hessian_mult method')

    if bool_indices:
        return (_hessian,) + tuple(parts)
    else:
        return _hessian

def _pearsonX2(y,
               linpred,
               loglike,
               df_fit):

    W = loglike.saturated_loss.hessian(linpred)
    n = y.shape[0]
    resid = y - loglike.saturated_loss.mean_function(linpred)
    return (resid ** 2 / W).sum() / (n - df_fit)
