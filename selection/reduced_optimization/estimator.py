import numpy as np
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov

class M_estimator_approx(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer

    def solve_approx(self):

        self.solve()

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate((_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)

        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform

        self._score_linear_term = np.concatenate((_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        self.feasible_point = np.append(self.observed_score_state, np.abs(self.initial_soln[self._overall]))

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
        self.score_cov = score_cov

        nactive = self._overall.sum()
        self.nactive = nactive

        self.B = self._opt_linear_term
        self.A = self._score_linear_term

        self.B_active = self.B[:nactive, :nactive]
        self.B_inactive = self.B[nactive:, :nactive]

        self.A_active = self._score_linear_term[:nactive, :]
        self.A_inactive = self._score_linear_term[nactive:, :]

        self.offset_active = self._opt_affine_term[:nactive]





