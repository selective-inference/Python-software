import functools
import numpy as np
import regreg.api as rr

from .query import query
from .cv import CV
from .cv_glmnet import CV_glmnet, have_glmnet
from .glm import bootstrap_cov
from .randomization import randomization

class CV_view(query):

    def __init__(self, glm_loss, loss_label, lasso_randomization=None, epsilon=None,  scale1=None, scale2=None):

        self.loss = glm_loss
        self.loss_label = loss_label
        self.lasso_randomization=lasso_randomization
        self.epsilon = epsilon
        self.scale1 = scale1
        self.scale2 = scale2
        X, _ = glm_loss.data
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.nboot = 1000

    def solve(self, glmnet=False, K=5):

        if glmnet == False:
            X, y = self.loss.data
            n, p = X.shape
            if self.loss_label == "gaussian":
                lam_seq = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))) +
                                          self.lasso_randomization.sample((1000,))).max(0))
            elif self.loss_label == 'logistic':
                lam_seq = np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 1000))) +\
                          self.lasso_randomization.sample((1000,))).max(0))
            self.lam_seq = np.exp(np.linspace(np.log(1.e-3), np.log(1), 30)) * lam_seq

            folds = np.arange(n) % K
            np.random.shuffle(folds)
            CV_compute = CV(self.loss,
                            folds,
                            self.lam_seq,
                            objective_randomization=self.lasso_randomization,
                            epsilon=self.epsilon)
        else:
            CV_compute = CV_glmnet(self.loss, self.loss_label)

        self.lam_CVR, self.SD, CVR_val, CV1_val, self.lam_seq = CV_compute.choose_lambda_CVR(self.scale1, self.scale2)
        self.ndim = self.lam_seq.shape[0]

        if (self.scale1 is not None) and (self.scale2 is not None):
            self.SD = self.SD+self.scale1**2+self.scale2**2
        (self.observed_opt_state, self.observed_score_state) = (CVR_val, CV1_val)
        self.num_opt_var = self.lam_seq.shape[0]
        self.lam_idx = list(self.lam_seq).index(self.lam_CVR)  # index of the minimizer

        self.opt_transform = (np.identity(self.num_opt_var), np.zeros(self.num_opt_var))
        self.score_transform = (-np.identity(self.num_opt_var), np.zeros(self.num_opt_var))

        self._marginalize_subgradient = False
        if self.scale1 is not None:
            self.randomization1 = randomization.isotropic_gaussian((self.num_opt_var,), scale=self.scale1)
            self.randomization2 = randomization.isotropic_gaussian((self.num_opt_var,), scale=self.scale2)
            query.__init__(self, self.randomization2)
            self.CVR_boot, self.CV1_boot = CV_compute.bootstrap_CVR_curve(self.scale1, self.scale2)
            self._solved = True

    def setup_sampler(self):
        return self.CV1_boot

    def one_SD_rule(self, direction="up"):
        CVR_val = self.observed_opt_state
        minimum_CVR = np.min(CVR_val)
        #CVR_cov = bootstrap_cov(lambda: np.random.choice(self.n, size=(self.n,), replace=True), self.CVR_boot, nsample=2)
        #SD = np.sqrt(np.diag(CVR_cov))

        SD_min = self.SD[self.lam_idx]
        #in glment lam_seq is decreasing
        if direction=="down":
            lam_1SD = self.lam_seq[max([i for i in range(self.lam_seq.shape[0]) if CVR_val[i] <= minimum_CVR+SD_min])]
        else:
            lam_1SD = self.lam_seq[min([i for i in range(self.lam_seq.shape[0]) if CVR_val[i] <= minimum_CVR+SD_min])]

        return lam_1SD

    def projection(self, opt_state):
        if self.opt_transform[0] is not None:
            return projection(opt_state, self.lam_idx)
        return None

    def condition_on_opt_state(self):
        self.num_opt_var = 0
        self.opt_transform = (None, self.observed_opt_state)


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
    else:
        val = np.mean(Z)

    opt = np.maximum(Z, val)
    opt[idx] = val
    return opt
