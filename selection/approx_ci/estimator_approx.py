import numpy as np
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov

from selection.randomized.threshold_score import threshold_score
from selection.randomized.greedy_step import greedy_score_step

class M_estimator_approx(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer

    def solve_approx(self):
        self.solve()
        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self.feasible_point = np.abs(self.initial_soln[self._overall])
        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)
        nactive = self._overall.sum()
        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self._opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]


class threshold_score_approx(threshold_score):

    def __init__(self, loss,
                 threshold,
                 randomization,
                 active_bool,
                 inactive_bool,
                 randomizer):

        threshold_score.__init__(self, loss, threshold, randomization, active_bool, inactive_bool)
        self.randomizer = randomizer

    def solve_approx(self):
        self.solve()
        self.setup_sampler()
        self.feasible_point = self.observed_opt_state[self.boundary]
        (_opt_linear_term, _opt_offset) = self.opt_transform
        self._opt_linear_term = np.concatenate((_opt_linear_term[self.boundary, :], _opt_linear_term[self.interior, :]),
                                               0)
        self._opt_affine_term = np.concatenate((_opt_offset[self.boundary], _opt_offset[self.interior]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self.boundary, :], _score_linear_term[self.interior, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self._overall = self.boundary
        self.inactive_lagrange = self.threshold[0] * np.ones(np.sum(~self.boundary))

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)
        nactive = self._overall.sum()
        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self._opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]

class greedy_score_step_approx(greedy_score_step):

    def __init__(self, loss,
                 penalty,
                 active_groups,
                 inactive_groups,
                 randomization,
                 randomizer):

        greedy_score_step.__init__(self, loss,
                                 penalty,
                                 active_groups,
                                 inactive_groups,
                                 randomization)
        self.randomizer = randomizer


    def solve_approx(self):

        self.solve()
        self.setup_sampler()
        p = self.inactive.sum()
        self.feasible_point = self.observed_scaling
        self._overall = np.zeros(p, dtype=bool)
        #print(self.selection_variable['variables'])
        self._overall[self.selection_variable['variables']] = 1

        self.observed_opt_state = np.hstack([self.observed_scaling, self.observed_subgradients])

        _opt_linear_term = np.concatenate((np.atleast_2d(self.maximizing_subgrad).T, self.losing_padding_map), 1)
        self._opt_linear_term = np.concatenate((_opt_linear_term[self._overall,:], _opt_linear_term[~self._overall,:]), 0)

        self.opt_transform = (self._opt_linear_term, np.zeros(p))

        (self._score_linear_term, _) = self.score_transform

        self.inactive_lagrange = self.observed_scaling * self.penalty.weights[0] * np.ones(p-1)

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self.active,
                                              inactive=~self.active)[0]

        bootstrap_target, target_observed = pairs_bootstrap_glm(self.loss,
                                                             self._overall,
                                                             beta_full=None,
                                                             inactive=None)

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)
        self.target_cov, target_score_cov = bootstrap_cov(sampler, bootstrap_target, cross_terms=(bootstrap_score,))
        self.score_target_cov = np.atleast_2d(target_score_cov).T
        self.target_observed = target_observed

        nactive = self._overall.sum()
        self.nactive = nactive

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]