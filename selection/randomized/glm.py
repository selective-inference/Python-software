import numpy as np

from .M_estimator import restricted_Mest, M_estimator
from .greedy_step import greedy_score_step
from regreg.api import glm

def pairs_bootstrap_glm(glm_loss, 
                        active, 
                        beta_full=None, 
                        inactive=None, 
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

    nactive = active.sum()
    if inactive is not None:
        X_full = np.hstack([X_active,X_inactive])
        beta_overall = np.zeros(X_full.shape[1])
        beta_overall[:nactive] = beta_active
    else:
        X_full = X_active
        beta_overall = beta_active

    _boot_mu = lambda X_full: glm_loss.saturated_loss.smooth_objective(X_full.dot(beta_overall), 'grad') + Y

    if ntotal > nactive:
        observed = np.hstack([beta_active, -glm_loss.smooth_objective(beta_full, 'grad')[inactive]])
    else:
        observed = beta_active

    def _boot_score(indices):
        X_star = X_full[indices]
        Y_star = Y[indices]
        score = X_star.T.dot(Y_star - _boot_mu(X_star))
        result = np.zeros(ntotal)
        result[:nactive] = _bootQinv.dot(score[:nactive])
        if ntotal > nactive:
            result[nactive:] = score[nactive:] + _bootI.dot(result[:nactive])
        return result

    return _boot_score, observed



def _parametric_cov_glm(glm_loss,
                        active,
                        beta_full=None,
                        inactive=None,
                        solve_args={'min_its': 50, 'tol': 1.e-10}):
    X, Y = glm_loss.data
    n, p = X.shape

    if beta_full is None:
        beta_active = restricted_Mest(glm_loss, active, solve_args=solve_args)
        beta_full = np.zeros(glm_loss.shape)
        beta_full[active] = beta_active
    else:
        beta_active = beta_full[active]

    X_active = X[:, active]

    nactive = active.sum()
    ntotal = nactive

    if inactive is not None:
        X_inactive = X[:, inactive]
        ntotal += inactive.sum()

    _bootW = np.diag(glm_loss.saturated_loss.hessian(X_active.dot(beta_active)))
    _bootQ = X_active.T.dot(_bootW.dot(X_active))
    _bootQinv = np.linalg.inv(_bootQ)
    if inactive is not None:
        _bootC = X_inactive.T.dot(_bootW.dot(X_active))
        _bootI = _bootC.dot(_bootQinv)

    nactive = active.sum()
    if inactive is not None:
        X_full = np.hstack([X_active, X_inactive])
        beta_overall = np.zeros(X_full.shape[1])
        beta_overall[:nactive] = beta_active
    else:
        X_full = X_active
        beta_overall = beta_active

    mat = np.zeros((p, n))
    mat[:nactive, :] = _bootQinv.dot(X_active.T)
    if ntotal>nactive:
        mat1 = np.dot(np.dot(_bootW, X_active), np.dot(_bootQinv, X_active.T))
        mat[nactive:, :] = X[:, inactive].T.dot(np.identity(n) - mat1)

    Sigma_full = np.dot(mat, np.dot(_bootW, mat.T))
    return Sigma_full


def _parametric_cov(glm_loss,
                    active_union, active_individual,
                    target_indices, param_cov, cross_terms,
                    solve_args={'min_its': 50, 'tol': 1.e-10}):

    parametric_cov = []
    parametric_cov.append(param_cov[target_indices,:][:,target_indices])

    X, Y = glm_loss.data
    n, p = X.shape
    def _WQ(active):
        beta_active = restricted_Mest(glm_loss, active, solve_args=solve_args)
        W = np.diag(glm_loss.saturated_loss.hessian(X[:,active].dot(beta_active)))
        Q = np.dot(X[:, active].T, W.dot(X[:, active]))
        Q_inv = np.linalg.inv(Q)

        return W, Q, Q_inv, Q_inv.dot(X[:, active].T)

    W_E, Q_E, Q_E_inv, mat_E = _WQ(active_union)

    for active in active_individual:
        W_E1, Q_E1, Q_E1_inv, mat_E1 = _WQ(active)
        C_E1 = np.dot(X[:, ~active].T, W_E1.dot(X[:, active]))
        I_E1 = C_E1.dot(Q_E1_inv)
        null_E1 = X[:, ~active].T - np.dot(I_E1, X[:, active].T)

        second_term = np.concatenate((mat_E1, null_E1), axis=0)
        cov = np.dot(mat_E, W_E.dot(second_term.T))
        parametric_cov.append(cov[target_indices, :])

    return parametric_cov




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
    return [_cov_target] + [_o - np.multiply.outer(_mean_target, _m) for _m, _o in zip(_mean_cross, _outer_cross)]


def pairs_inactive_score_glm(glm_loss, active, beta_active):

    """
    Bootstrap inactive score at \bar{\beta}_E

    Will be used with forward stepwise.
    """
    inactive = ~active
    beta_full = np.zeros(glm_loss.shape)
    beta_full[active] = beta_active

    _full_boot_score = pairs_bootstrap_glm(glm_loss, 
                                           active, 
                                           beta_full=beta_full,
                                           inactive=inactive)[0]
    nactive = active.sum()
    def _boot_score(indices):
        return _full_boot_score(indices)[nactive:]

    return _boot_score

class glm_group_lasso(M_estimator):

    def setup_sampler(self):
        M_estimator.setup_sampler(self)

        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self.overall, 
                                              beta_full=self._beta_full,
                                              inactive=self.inactive)[0]

        parametric_cov = _parametric_cov_glm(self.loss,
                                             self.overall,
                                             beta_full=self._beta_full,
                                             inactive=self.inactive)

        return bootstrap_score, parametric_cov


class glm_greedy_step(greedy_score_step):

    def setup_sampler(self):
        greedy_score_step.setup_sampler(self)

        bootstrap_score = pairs_inactive_score_glm(self.loss, 
                                                   self.active,
                                                   self.beta_active)
        return bootstrap_score

def resid_bootstrap(gaussian_loss,
                    active,
                    inactive=None):

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

    def _boot_score(Y_star):
        beta_hat = X_active_inv.dot(Y_star)
        result = np.zeros(ntotal)
        result[:nactive] = beta_hat
        if ntotal > nactive:
            result[nactive:] = X_inactive_resid.T.dot(Y_star)
        return result

    return _boot_score, observed


class fixedX_group_lasso(M_estimator):

    def __init__(self, X, Y, epsilon, penalty, randomization, solve_args={'min_its':50, 'tol':1.e-10}):
        loss = glm.gaussian(X, Y)
        M_estimator.__init__(self,
                             loss, 
                             epsilon, 
                             penalty, 
                             randomization, solve_args=solve_args)

    def setup_sampler(self):
        M_estimator.setup_sampler(self)

        X, Y = self.loss.data

        bootstrap_score = resid_bootstrap(self.loss,
                                          self.overall, 
                                          self.inactive)[0]
        return bootstrap_score
