import functools
import numpy as np
import regreg.api as rr
from .query import query
from selection.randomized.cv import CV
from selection.randomized.glm import bootstrap_cov
from selection.api import randomization

class CV_view(query):

    def __init__(self, glm_loss, lasso_randomization, epsilon, loss, scale1=0.1, scale2=0.1, K=5):

        self.loss = glm_loss
        X, y = self.loss.data
        n, p = X.shape
        if loss=="gaussian":
            #lam_seq = np.mean(np.fabs(np.dot(X.T, y)))
            lam_seq = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))+lasso_randomization.sample((1000,))).max(0))
        elif loss=='logistic':
            lam_seq = np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 1000)))+lasso_randomization.sample((1000,))).max(0))
        lam_seq = np.exp(np.linspace(np.log(1.e-6), np.log(1), 30)) * lam_seq
        # lam_seq = np.exp(np.linspace(np.log(1.e-2), np.log(2), 30)) * np.fabs(X.T.dot(y)+lasso_randomization.sample((10,))).max()

        folds = np.arange(n) % K
        np.random.shuffle(folds)
        (self.folds,
         self.lam_seq,
         self.lasso_randomization,
         self.epsilon,
         self.n,
         self.p,
         self.K) = (folds,
                    lam_seq,
                    lasso_randomization,
                    epsilon,
                    n,
                    p,
                    K)
        self.num_opt_var = self.lam_seq.shape[0]
        self.randomization1 = randomization.isotropic_gaussian((self.num_opt_var,), scale=scale1)
        self.randomization2 = randomization.isotropic_gaussian((self.num_opt_var,), scale=scale2)
        query.__init__(self, self.randomization2)
        self.nboot = 1
        self.scale1 = scale1
        self.scale2 = scale2

    def solve(self):

        CV_compute = CV(self.loss,
                        self.folds,
                        self.lam_seq,
                        objective_randomization=self.lasso_randomization,
                        epsilon=self.epsilon)

        self.lam_CVR, SD, CVR_val, CV1_val = CV_compute.choose_lambda_CVR(self.randomization1, self.randomization2)
        self.SD = SD +self.scale1**2+self.scale2**2
        (self.observed_opt_state, self.observed_score_state) = (CVR_val, CV1_val)

        self.lam_idx = list(self.lam_seq).index(self.lam_CVR) # index of the minimizer
        print("index of the minimizer", self.lam_idx)
        self._solved = True
        self.opt_transform = (np.identity(self.num_opt_var), np.zeros(self.num_opt_var))
        self.score_transform = (-np.identity(self.num_opt_var), np.zeros(self.num_opt_var))

        self._marginalize_subgradient = False

        self.CVR_boot, self.CV1_boot = CV_compute.bootstrap_CVR_curve(self.randomization1, self.randomization2)

    def setup_sampler(self):
        return self.CV1_boot

    def one_SD_rule(self):
        CVR_val = self.observed_opt_state
        minimum_CVR = np.min(CVR_val)
        #CVR_cov = bootstrap_cov(lambda: np.random.choice(self.n, size=(self.n,), replace=True), self.CVR_boot, nsample=2)
        #SD = np.sqrt(np.diag(CVR_cov))
        #print("SD vector", SD)
        #print("CVR_val", CVR_val)
        #lam_1SD = self.lam_seq[max([i for i in range(self.lam_seq.shape[0]) if CVR_val[i] <= 1.05*minimum_CVR])]
        #lam_1SD = self.lam_seq[min([i for i in range(self.lam_seq.shape[0]) if CVR_val[i] <= 1.05*minimum_CVR])]
        #print(0.05*minimum_CVR, self.SD)
        gap = np.mean(self.SD)
        #lam_1SD = self.lam_seq[min([i for i in range(self.lam_seq.shape[0]) if CVR_val[i] <= minimum_CVR+SD[i]])]
        lam_1SD = self.lam_seq[min([i for i in range(self.lam_seq.shape[0]) if CVR_val[i] <= minimum_CVR+gap])]
        return lam_1SD

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
