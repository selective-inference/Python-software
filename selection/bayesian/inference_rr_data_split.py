import time
import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm
from selection.randomized.M_estimator import M_estimator, M_estimator_split
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov
from selection.tests.instance import logistic_instance, gaussian_instance

class inputs(M_estimator_split):

    def __init__(self, loss, epsilon, penalty, coef=1., offset=None, quadratic=None, nstep=10):

        total_size = loss.saturated_loss.shape[0]

        subsample_size = int(0.8 * total_size)

        M_estimator_split.__init__(self, loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10})


    def solve_approx(self):

        self.Msolve()
        randomization_cov = self.setup_sampler()
        print(randomization_cov.shape)
        X, _ = self.loss.data
        n, p = X.shape
        self.feasible_point = np.abs(self.initial_soln[self._overall])
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        score_linear_term = self.score_transform[0]
        (self.opt_linear_term, self.opt_affine_term) = self.opt_transform

        self.score_linear_term = score_linear_term

p=10
n= 100
X, y, beta, nonzero, sigma = gaussian_instance(n=100, p=10, s=0, rho=0., snr=5, sigma=1.)
lam_frac=1.
lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
loss = rr.glm.gaussian(X, y)

epsilon = 1. / np.sqrt(n)
W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p),weights=dict(zip(np.arange(p), W)), lagrange=1.)
test = inputs(loss, epsilon, penalty)
test.solve_approx()



