import functools
import numpy as np
import regreg.api as rr
from .query import query
from selection.randomized.cv import (choose_lambda_CV,
                                     bootstrap_CV_curve)
from selection.randomized.glm import (pairs_bootstrap_glm,
                                      glm_nonparametric_bootstrap)

from selection.api import randomization


class CV_view(query):

    def __init__(self, loss, scale1=0.5, scale2=0.5, K=5):

        self.loss = loss
        X, y = loss.data
        n, p = X.shape
        lam_seq = np.exp(np.linspace(np.log(1.e-2), np.log(1), 30)) * np.fabs(X.T.dot(y)).max()
        folds = np.arange(n) % K
        np.random.shuffle(folds)
        (self.folds,
         self.lam_seq,
         self.n,
         self.p,
         self.K) = (folds,
                    lam_seq,
                    n,
                    p,
                    K)

        self.num_opt_var = self.lam_seq.shape[0]
        self.randomization1 = randomization.isotropic_gaussian((self.num_opt_var,), scale=scale1)
        self.randomization2 = randomization.isotropic_gaussian((self.num_opt_var,), scale=scale2)
        query.__init__(self, self.randomization2)
        self.nboot = 1

    def solve(self, scale1=0.5, scale2=0.5):


        lam_CVR, CVR_val, CV1_val = choose_lambda_CV(self.loss, self.lam_seq, self.folds, self.randomization1, self.randomization2)

        (self.lam_CVR,
         self.observed_opt_state,
         self.observed_score_state) = (lam_CVR,
                                       CVR_val,
                                       CV1_val)
        self.lam_idx = list(self.lam_seq).index(self.lam_CVR) # index of the minimizer
        self._solved = True
        self.projection_map = projection_cone(self.num_opt_var, self.lam_idx,1)
        self.opt_transform = (np.identity(self.num_opt_var), np.zeros(self.num_opt_var))
        self.score_transform = (-np.identity(self.num_opt_var), np.zeros(self.num_opt_var))

        self._marginalize_subgradient = False

    def setup_sampler(self):
        CV1_boot = bootstrap_CV_curve(self.loss, self.lam_seq, self.folds, self.K, self.randomization1, self.randomization2)
        return CV1_boot

    def projection(self, opt_state):
        return -self.projection_map(-opt_state)


    def condition_on_opt_state(self):
        self.num_opt_var = 0
        self.opt_transform = (None, self.observed_opt_state)


def projection_cone(p, max_idx, max_sign):
    """
    Create a callable that projects onto one of two cones,
    determined by which coordinate achieves the max in one
    step of forward stepwise.
    Parameters
    ----------
    p : int
        Dimension.
    max_idx : int
        Index achieving the max.
    max_sign : [-1,1]
        Sign of achieved max.
    Returns
    -------
    projection : callable
        A function to compute projection onto appropriate cone.
        Takes one argument of shape (p,).
    """

    if max_sign > 0:
        P = rr.linf_epigraph(p-1)
    else:
        P = rr.l1_epigraph_polar(p-1)

    def _projection(state):
        permuted_state = np.zeros_like(state)
        permuted_state[-1] = state[max_idx]
        permuted_state[:max_idx] = state[:max_idx]
        permuted_state[max_idx:-1] = state[(max_idx+1):]

        projected_state = P.cone_prox(permuted_state)

        new_state = np.zeros_like(state)
        new_state[max_idx] = projected_state[-1]
        new_state[:max_idx] = projected_state[:max_idx]
        new_state[(max_idx+1):] = projected_state[max_idx:-1]

        return new_state

    return _projection
