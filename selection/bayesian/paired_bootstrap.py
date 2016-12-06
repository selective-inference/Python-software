import functools # for bootstrap partial mapping

import numpy as np
from selection.randomized.M_estimator import restricted_Mest, M_estimator

def pairs_bootstrap_glm(glm_loss,
                        active,
                        beta_full=None,
                        inactive=None,
                        scaling=1.,
                        solve_args={'min_its':50, 'tol':1.e-10}):
    """
    pairs bootstrap of (beta_hat_active, -grad_inactive(beta_hat_active))
    """
    X, Y = glm_loss.data

    if beta_full is None:
        beta_active = restricted_Mest(glm_loss, active, solve_args=solve_args)
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
        X_full = np.hstack([X_active,X_inactive])
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

    for _ in range(nsample):
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

