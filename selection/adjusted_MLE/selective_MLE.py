import numpy as np
import functools
from selection.randomized.M_estimator import M_estimator

class M_estimator_map(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomization_scale = 1.):
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

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        self.randomizer_precision = (1. / self.randomization_scale) * np.identity(p)

        score_cov = np.zeros((p, p))
        X_active_inv = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
        projection_perp = np.identity(n) - X[:, self._overall].dot(X_active_inv).dot(X[:, self._overall].T)
        score_cov[:self.nactive, :self.nactive] = X_active_inv
        score_cov[self.nactive:, self.nactive:] = X[:, ~self._overall].T.dot(projection_perp).dot(X[:, ~self._overall])
        self.score_cov = score_cov
        self.observed_score_state = self.observed_internal_state
        self.target_observed = self.observed_internal_state[:self.nactive]
        self.score_target_cov = self.score_cov[:, :self.nactive]
        self.target_cov = self.score_cov[:self.nactive, :self.nactive]

    def solve_map(self):
        self.feasible_point = np.abs(self.initial_soln[self._overall])

        self.A = np.dot(self._score_linear_term, self.score_target_cov).dot(np.linalg.inv(self.target_cov))
        self.data_offset = self._score_linear_term.dot(self.observed_score_state)- self.A.dot(self.target_observed)
        self.target_transform = (self.A, self.data_offset)

    def solve_map_univariate_target(self, j):
        self.feasible_point = np.abs(self.initial_soln[self._overall])[j]

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.data_offset = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]
        self.target_transform = (self.A.reshape((self.A.shape[0],1)),
                                 self.data_offset.reshape((self.data_offset.shape[0],1)))


def solve_UMVU(target_transform,
               opt_transform,
               target_observed,
               feasible_point,
               target_cov,
               randomizer_precision,
               step=1,
               nstep=30,
               tol=1.e-8):

    A, data_offset = target_transform # data_offset = N
    B, opt_offset = opt_transform     # opt_offset = u

    nopt = B.shape[1]
    ntarget = A.shape[1]

    #assert ntarget == 1

    # setup joint implied covariance matrix
    if ntarget>1:
        target_precision = np.linalg.inv(target_cov)
    else:
        target_precision = 1./target_cov
        opt_offset = opt_offset.reshape((opt_offset.shape[0],1))

    implied_precision = np.zeros((ntarget + nopt, ntarget + nopt))

    #print("shapes", A.shape, (A.T.dot(randomizer_precision).dot(A)).shape, target_precision.shape)
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
    #print("shapes", data_offset.shape, opt_offset.shape, conditioned_value.shape)

    linear_term = implied_cross.T.dot(np.linalg.inv(implied_target))
    offset_term = -B.T.dot(randomizer_precision).dot(conditioned_value)
    #print("check shapes", linear_term.dot(target_observed).shape, offset_term.shape)
    natparam_transform = (linear_term, offset_term)
    conditional_natural_parameter = linear_term.dot(target_observed) + offset_term

    conditional_precision = implied_precision[ntarget:,ntarget:]

    #print("check shapes", conditional_natural_parameter.shape, conditional_precision.shape)
    soln, value = solve_barrier_nonneg(conditional_natural_parameter,
                                       conditional_precision,
                                       feasible_point=feasible_point)
    M_1_inv = np.linalg.inv(M_1)
    offset_term = - M_1_inv.dot(M_2.dot(conditioned_value))
    linear_term = np.vstack([M_1_inv, -M_1_inv.dot(L)])
    mle_transform = (M_1_inv, -M_1_inv.dot(L), offset_term)

    def mle_map(natparam_transform, mle_transform, feasible_point, conditional_precision, target_observed):
        param_lin, param_offset = natparam_transform
        mle_target_lin, mle_soln_lin, mle_offset = mle_transform
        soln, value = solve_barrier_nonneg(param_lin.dot(target_observed) + param_offset,
                                           conditional_precision,
                                           feasible_point=feasible_point)
        return mle_target_lin.dot(target_observed) + mle_soln_lin.dot(soln) + mle_offset, value

    mle_partial = functools.partial(mle_map, natparam_transform, mle_transform, feasible_point, conditional_precision)
    sel_MLE, value = mle_partial(target_observed)
    return np.squeeze(sel_MLE), value, mle_partial


def solve_barrier_nonneg(conjugate_arg,
                         precision,
                         feasible_point=None,
                         step=1,
                         nstep=30,
                         tol=1.e-8):

    #conjugate_arg = precision.dot(mean_vec)
    scaling = np.sqrt(np.diag(precision))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. + np.log(1.+ 1./(u / scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) + (1./(scaling + u) - 1./u)

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

    return current, current_value






















