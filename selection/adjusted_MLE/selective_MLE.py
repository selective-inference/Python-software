import numpy as np
import regreg.api as rr
from selection.randomized.M_estimator import M_estimator

class M_estimator_map(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomization_scale = 1.):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomization
        self.randomization_scale = randomization_scale

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
        nactive = self._overall.sum()
        self.inactive_subgrad = self.observed_opt_state[nactive:]


        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p


        score_cov = np.zeros((p, p))
        X_active_inv = np.linalg.inv(X[:,self._overall].T.dot(X[:,self._overall]))
        projection_perp = np.identity(n) - X[:,self._overall].dot(X_active_inv).dot( X[:,self._overall].T)
        score_cov[:nactive, :nactive] = X_active_inv
        score_cov[nactive:, nactive:] = X[:,~self._overall].T.dot(projection_perp).dot(X[:,~self._overall])

        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_internal_state[:nactive]
        self.observed_score_state = self.observed_internal_state
        self.nactive = nactive

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]
        self.B = np.vstack([self.B_active, self.B_inactive])


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self._opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]

class selective_MLE():
    def __init__(self,
                 map):

        self.map = map
        self.randomizer_precision = map.randomizer.precision
        self.target_observed = self.map.target_observed
        self.nactive = self.target_observed.shape[0]
        self.target_cov = self.map.target_cov

        initial = self.map.feasible_point

    def solve_Gaussian_density(self, j):

        self.map.setup_map(j)
        inverse_cov = np.zeros((1+self.nactive, 1+self.nactive))
        inverse_cov[0,0] = self.map.A.T.dot(self.randomizer_precision).dot(self.map.A) + 1./self.target_cov[j,j]
        inverse_cov[0,0:] = self.map.A.T.dot(self.randomizer_precision).dot(self.map.B)
        inverse_cov[0:,0] = self.map.B.T.dot(self.randomizer_precision).self.map.A
        inverse_cov[0:,0:] = self.map.B.T.dot(self.randomizer_precision).self.map.B
        cov = np.linalg.inv(inverse_cov)

        self.L = cov[0,0:].dot(np.linalg.inv(cov[0:,0:]))
        self.M_1 = (1./inverse_cov[0,0])*(1./self.target_cov[j,j])
        self.M_2 = (1./inverse_cov[0,0]).dot(self.map.A.T).dot(self.randomizer_precision)
        self.inactive_subgrad = np.zeros(self.map.p)
        self.inactive_subgrad[self.nactive:] = self.map.inactive_subgrad
        self.conditioned_value = self.map.null_statistic + self.map.inactive_subgrad

        self.conditional_par = inverse_cov[0:,0:].dot(cov[0:,0]).dot((1./cov[0,0])* self.target_observed[j]) + \
                               self.B.T(self.randomizer_precision).dot(self.conditioned_value)
        self.conditional_var = inverse_cov[0:,0:]

    def solve_UMVU(self, j, step=1, nstep=30, tol=1.e-8):

        objective = lambda u: u.T.dot(self.conditional_par) - u.T.dot(self.conditional_var).dot(u)/2. - np.log(1.+ 1./u)
        grad = lambda u: self.conditional_par - self.conditional_var.dot(u) - 1./(1.+ u) + 1./u

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
                # print(current_value, proposed_value, 'minimize')
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

                # print('iter', itercount)
        value = objective(current)
        return -(1./self.M_1)*self.L.dot(current)+ (1./self.M_1)*(self.target_observed[j]- self.M_2.dot(self.conditioned_value)), \
               value























