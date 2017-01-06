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

    def __init__(self, loss, scale1=0.1, scale2=0.5, K=5):

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
        #print(self.randomization1.sample())
        self.nboot = 1

    def solve(self):


        lam_CVR, CVR_val, CV1_val = choose_lambda_CV(self.loss, self.lam_seq, self.folds, self.randomization1, self.randomization2)

        (self.lam_CVR,
         self.observed_opt_state,
         self.observed_score_state) = (lam_CVR,
                                       CVR_val,
                                       CV1_val)
        self.lam_idx = list(self.lam_seq).index(self.lam_CVR) # index of the minimizer
        self._solved = True
        self.opt_transform = (np.identity(self.num_opt_var), np.zeros(self.num_opt_var))
        self.score_transform = (-np.identity(self.num_opt_var), np.zeros(self.num_opt_var))

        self._marginalize_subgradient = False

    def setup_sampler(self):
        CV1_boot = bootstrap_CV_curve(self.loss, self.lam_seq, self.folds, self.K, self.randomization1, self.randomization2)
        return CV1_boot

    def projection(self, opt_state):
        if self.opt_transform[0] is not None:
            return projection(opt_state, self.lam_idx)
        return None

    def condition_on_opt_state(self):
        self.num_opt_var = 0
        self.opt_transform = (None, self.observed_opt_state)



#DEBUG = True
def projection(Z, idx):
    Z = np.asarray(Z)
    keep = np.ones_like(Z, np.bool)
    keep[idx] = 0
    Z_sort = np.sort(Z[keep])

    if np.all(Z[keep] >= Z[idx]):
        return Z

    root_found = False
    for i in range(Z_sort.shape[0] - 1):
        left_val = Z_sort[i] - Z[idx] + np.sum(keep * (Z <= Z_sort[i]) * (Z_sort[i] - Z))
        right_val = Z_sort[i+1] - Z[idx] + np.sum(keep * (Z <= Z_sort[i+1]) * (Z_sort[i+1] - Z))
        if left_val * right_val < 0:
            root_found = True
            break

    if root_found:
        val = (np.sum(Z_sort[:(i+1)]) + Z[idx]) / (i+2)
        dval = val - Z[idx] + np.sum(keep * (Z <= val) * (val - Z))
        #if DEBUG:
        #    print('derivative is:', dval)
    else:
        val = np.mean(Z)

    opt = np.maximum(Z, val)
    opt[idx] = val
    return opt
