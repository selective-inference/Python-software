import numpy as np
from selection.randomized.M_estimator import M_estimator, M_estimator_split
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov

class M_estimator_approx(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer, estimation):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer
        self.estimation = estimation

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
        nactive = self._overall.sum()
        self.nactive = nactive

        self.target_observed = self.observed_score_state[:self.nactive]

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))
            inv_X_active = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
            projection_X_active = X[:,self._overall].dot(np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))).dot(X[:,self._overall].T)
            score_cov[:self.nactive, :self.nactive] = inv_X_active
            score_cov[self.nactive:, self.nactive:] = X[:,~self._overall].T.dot(np.identity(n)- projection_X_active).dot(X[:,~self._overall])

        elif self.estimation == 'bootstrap':
            bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        self.score_cov = score_cov
        self.target_cov = score_cov[:nactive, :nactive]
        self.score_cov_inv = np.linalg.inv(self.score_cov)


        self.B = self._opt_linear_term
        self.A = self._score_linear_term

        self.B_active = self.B[:nactive, :nactive]
        self.B_inactive = self.B[nactive:, :nactive]

        self.A_active = self._score_linear_term[:nactive, :]
        self.A_inactive = self._score_linear_term[nactive:, :]

        self.offset_active = self._opt_affine_term[:nactive]


class M_estimator_approx_carved(M_estimator_split):

    def __init__(self, loss, epsilon, subsample_size, penalty, estimation):

        M_estimator_split.__init__(self,loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10})
        self.estimation = estimation

    def solve_approx(self):

        self.solve()

        self.nactive = self._overall.sum()
        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        self.target_observed = self.observed_score_state[:self.nactive]

        self.feasible_point = np.concatenate([self.observed_score_state, np.fabs(self.observed_opt_state[:self.nactive]),
                                              self.observed_opt_state[self.nactive:]], axis = 0)

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)

        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)

        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)

        #print("True or false", np.all(lagrange[0]-np.fabs(self.feasible_point[p+self.nactive:]))>0)
        #print("True or false", np.all(self.feasible_point[p:][:self.nactive]) > 0)

        self.inactive_lagrange = lagrange[~self._overall]

        self.bootstrap_score, self.randomization_cov = self.setup_sampler()

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))
            inv_X_active = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
            projection_X_active = X[:,self._overall].dot(np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))).dot(X[:,self._overall].T)
            score_cov[:self.nactive, :self.nactive] = inv_X_active
            score_cov[self.nactive:, self.nactive:] = X[:,~self._overall].T.dot(np.identity(n)- projection_X_active).dot(X[:,~self._overall])

        elif self.estimation == 'bootstrap':
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), self.bootstrap_score)

        self.score_cov = score_cov
        self.score_cov_inv = np.linalg.inv(self.score_cov)




class M_estimator_approx_logistic(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer, estimation):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer
        self.estimation = estimation

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
        nactive = self._overall.sum()
        self.nactive = nactive

        self.target_observed = self.observed_score_state[:self.nactive]

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))

            vec = np.exp(X[:, self._overall].dot(self.target_observed))
            #vec = np.exp(np.zeros(n))
            pi = np.true_divide(vec, np.power(1. + vec, 2))
            weights = np.diag(pi)
            Q_active = X[:, self._overall].T.dot(weights).dot(X[:, self._overall])
            Q_active_inv = np.linalg.inv(Q_active)

            P_inactive = X[:, ~self._overall].T.dot(np.identity(n)-weights.dot(X[:, self._overall].dot(Q_active_inv)
                                                                               .dot(X[:, self._overall].T)))
            score_cov[:self.nactive, :self.nactive] = Q_active_inv
            score_cov[self.nactive:, self.nactive:] = P_inactive.dot(weights).dot(P_inactive.T)

        elif self.estimation == 'bootstrap':
            bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        self.score_cov = score_cov
        self.target_cov = score_cov[:nactive, :nactive]
        self.score_cov_inv = np.linalg.inv(self.score_cov)


        self.B = self._opt_linear_term
        self.A = self._score_linear_term

        self.B_active = self.B[:nactive, :nactive]
        self.B_inactive = self.B[nactive:, :nactive]

        self.A_active = self._score_linear_term[:nactive, :]
        self.A_inactive = self._score_linear_term[nactive:, :]

        self.offset_active = self._opt_affine_term[:nactive]






