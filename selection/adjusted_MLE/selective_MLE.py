import numpy as np
import functools
from selection.randomized.M_estimator import M_estimator

class M_estimator_map(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, M, target="partial", randomization_scale = 1., sigma= 1.):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomization
        self.randomization_scale = randomization_scale

        self.solve()
        self.nactive = self._overall.sum()
        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall],
                                                _opt_affine_term[~self._overall] + self.observed_opt_state[self.nactive:]),
                                               0)
        self._opt_linear_term = self._opt_linear_term[:, :self._overall.sum()]
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)
        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        X, y = self.loss.data
        n, p = X.shape
        self.p = p

        self.randomizer_precision = (1. / self.randomization_scale) * np.identity(p)

        score_cov = np.zeros((p, p))
        X_active_inv = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
        projection_perp = np.identity(n) - X[:, self._overall].dot(X_active_inv).dot(X[:, self._overall].T)
        score_cov[:self.nactive, :self.nactive] = X_active_inv
        score_cov[self.nactive:, self.nactive:] = X[:, ~self._overall].T.dot(projection_perp).dot(X[:, ~self._overall])
        self.score_cov = (sigma**2.) * score_cov

        self.observed_score_state = self.observed_internal_state

        if self.nactive>0:
            if target == "partial":
                self.target_observed = self.observed_internal_state[:self.nactive]
                self.score_target_cov = self.score_cov[:, :self.nactive]
                self.target_cov = self.score_cov[:self.nactive, :self.nactive]
            elif target == 'full':
                X_full_inv = np.linalg.pinv(X)[self._overall]
                self.target_observed = X_full_inv.dot(y)  # unique to OLS!!!!
                self.target_cov = (sigma ** 2) * X_full_inv.dot(X_full_inv.T)
                self.score_target_cov = np.zeros((p, self.nactive))
                self.score_target_cov[:self.nactive] = np.linalg.pinv(X[:, self._overall]).dot(X_full_inv.T)
                self.score_target_cov[self.nactive:] = X[:, ~self._overall].T.dot(projection_perp.dot(X_full_inv.T))
                self.score_target_cov *= sigma ** 2
            elif target == 'debiased':
                X_full_inv = M.dot(X.T)[self._overall]
                self.target_observed = X_full_inv.dot(y)  # unique to OLS!!!!
                self.target_cov = (sigma ** 2) * X_full_inv.dot(X_full_inv.T)
                self.score_target_cov = np.zeros((p, self.nactive))
                self.score_target_cov[:self.nactive] = np.linalg.pinv(X[:, self._overall]).dot(X_full_inv.T)
                self.score_target_cov[self.nactive:] = X[:, ~self._overall].T.dot(projection_perp.dot(X_full_inv.T))
                self.score_target_cov *= sigma ** 2

    def solve_map(self):
        self.feasible_point = np.ones(self._overall.sum())
        self.A = np.dot(self._score_linear_term, self.score_target_cov).dot(np.linalg.inv(self.target_cov))
        self.data_offset = self._score_linear_term.dot(self.observed_score_state)- self.A.dot(self.target_observed)
        self.target_transform = (self.A, self.data_offset)

    def solve_map_univariate_target(self, j):
        self.feasible_point = np.ones(self._overall.sum())
        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.data_offset = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]
        self.target_transform = (self.A.reshape((self.A.shape[0],1)),self.data_offset)


def solve_UMVU(target_transform,
               opt_transform,
               target_observed,
               feasible_point,
               target_cov,
               randomizer_precision):

    A, data_offset = target_transform # data_offset = N
    B, opt_offset = opt_transform     # opt_offset = u

    nopt = B.shape[1]
    ntarget = A.shape[1]

    # setup joint implied covariance matrix

    target_precision = np.linalg.inv(target_cov)

    implied_precision = np.zeros((ntarget + nopt, ntarget + nopt))
    implied_precision[:ntarget,:ntarget] = A.T.dot(randomizer_precision).dot(A) + target_precision
    implied_precision[:ntarget,ntarget:] = A.T.dot(randomizer_precision).dot(B)
    implied_precision[ntarget:,:ntarget] = B.T.dot(randomizer_precision).dot(A)
    implied_precision[ntarget:,ntarget:] = B.T.dot(randomizer_precision).dot(B)
    implied_cov = np.linalg.inv(implied_precision)

    implied_opt = implied_cov[ntarget:,ntarget:]
    implied_target = implied_cov[:ntarget,:ntarget]
    implied_cross = implied_cov[:ntarget,ntarget:]

    L = implied_cross.dot(np.linalg.inv(implied_opt))
    M_1 = np.linalg.inv(implied_precision[:ntarget,:ntarget]).dot(target_precision)
    M_2 = -np.linalg.inv(implied_precision[:ntarget,:ntarget]).dot(A.T.dot(randomizer_precision))

    conditioned_value = data_offset + opt_offset

    linear_term = implied_precision[ntarget:,ntarget:].dot(implied_cross.T.dot(np.linalg.inv(implied_target)))
    offset_term = -B.T.dot(randomizer_precision).dot(conditioned_value)

    natparam_transform = (linear_term, offset_term)
    conditional_natural_parameter = linear_term.dot(target_observed) + offset_term

    conditional_precision = implied_precision[ntarget:,ntarget:]

    M_1_inv = np.linalg.inv(M_1)
    mle_offset_term = - M_1_inv.dot(M_2.dot(conditioned_value))
    mle_transform = (M_1_inv, -M_1_inv.dot(L), mle_offset_term)
    var_transform = (-implied_precision[ntarget:,:ntarget].dot(M_1),
                     -implied_precision[ntarget:,:ntarget].dot(M_2.dot(conditioned_value)))

    cross_covariance = np.linalg.inv(implied_precision[:ntarget, :ntarget]).dot(implied_precision[:ntarget, ntarget:])
    var_matrices = (np.linalg.inv(implied_opt), np.linalg.inv(implied_precision[:ntarget,:ntarget]),
                    cross_covariance,target_precision)

    def mle_map(natparam_transform, mle_transform, var_transform, var_matrices,
                feasible_point, conditional_precision, target_observed):

        param_lin, param_offset = natparam_transform
        mle_target_lin, mle_soln_lin, mle_offset = mle_transform

        soln, value, _ = solve_barrier_nonneg(param_lin.dot(target_observed) + param_offset,
                                              conditional_precision,
                                              feasible_point=feasible_point,
                                              step=1,
                                              nstep=2000,
                                              tol=1.e-8)

        selective_MLE = mle_target_lin.dot(target_observed) + mle_soln_lin.dot(soln) + mle_offset

        var_target_lin, var_offset = var_transform
        var_precision, inv_precision_target, cross_covariance, target_precision =  var_matrices
        _, _, hess = solve_barrier_nonneg(var_target_lin.dot(selective_MLE) + var_offset + mle_offset,
                                          var_precision,
                                          feasible_point=None,
                                          step=1,
                                          nstep=2000)

        hessian = target_precision.dot(inv_precision_target +
                                       cross_covariance.dot(hess).dot(cross_covariance.T)).dot(target_precision)

        return selective_MLE, np.linalg.inv(hessian)

    mle_partial = functools.partial(mle_map, natparam_transform, mle_transform, var_transform, var_matrices,
                                    feasible_point, conditional_precision)
    sel_MLE, inv_hessian = mle_partial(target_observed)

    implied_parameter = np.hstack([target_precision.dot(sel_MLE)-A.T.dot(randomizer_precision).dot(conditioned_value), offset_term])

    return np.squeeze(sel_MLE), inv_hessian, mle_partial, implied_cov, implied_cov.dot(implied_parameter), mle_transform

def solve_barrier_nonneg(conjugate_arg,
                         precision,
                         feasible_point=None,
                         step=1,
                         nstep=1000,
                         tol=1.e-8):

    scaling = np.sqrt(np.diag(precision))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. + np.log(1.+ 1./(u / scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) + (1./(scaling + u) - 1./u)
    barrier_hessian = lambda u: (-1./((scaling + u)**2.) + 1./(u**2.))

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        newton_step = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * newton_step
            if np.all(proposal > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            proposal = current - step * newton_step
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + np.diag(barrier_hessian(current)))
    return current, current_value, hess























