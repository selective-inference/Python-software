import numpy as np
import regreg.api as rr
#from selection.bayesian.cisEQTLS.tests.CV_lambda import tuning_parameter_glmnet
# from rpy2.robjects.packages import importr
# from rpy2 import robjects
# glmnet = importr('glmnet')
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
import numpy as np
import regreg.api as rr
from selection.tests.instance import gaussian_instance


# def tuning_parameter_glmnet(X, y):
#     robjects.r('''
#         glmnet_cv = function(X,y, lam_seq=NA){
#         y = as.matrix(y)
#         X = as.matrix(X)
#         if (is.na(lam_seq)){
#             G_CV = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
#         }
#         else {
#             G_CV = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE, lambda=lam_seq)
#         }
#         lam_1SE = G_CV$lambda.1se
#         lam_minCV = G_CV$lambda.min
#         n = nrow(X)
#         lam_minCV = lam_minCV*n
#         lam_1SE = lam_1SE*n
#         lam_seq = G_CV$lambda*n
#         result = list(lam_minCV=lam_minCV, lam_1SE=lam_1SE, lam_seq = lam_seq, CV_err=G_CV$cvm, SD=G_CV$cvsd)
#         return(result)
#         }''')
#
#     r_glmnet_cv = robjects.globalenv['glmnet_cv']
#     n, p = X.shape
#     r_X = robjects.r.matrix(X, nrow=n, ncol=p)
#     r_y = robjects.r.matrix(y, nrow=n, ncol=1)
#     result = r_glmnet_cv(r_X, r_y)
#     lam_minCV = result[0][0]
#     lam_1SE = result[1][0]
#     return lam_minCV, lam_1SE


def selection(X, y, random_Z, randomization_scale=1, sigma=None, method="theoretical"):
    n, p = X.shape
    loss = rr.glm.gaussian(X,y)
    epsilon = 1. / np.sqrt(n)
    lam_frac = 1.
    if sigma is None:
        sigma = 1.
    if method == "theoretical":
        lam = 1. * sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))
    # elif method == "cross-validation":
    #     lam = tuning_parameter_glmnet(X, y)[1]
    #     print(lam)

    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p), weights = dict(zip(np.arange(p), W)), lagrange=1.)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, -randomization_scale * random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}


    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    active = (initial_soln != 0)
    if np.sum(active) == 0:
        return None
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE = initial_soln[active]
    subgradient = -(initial_grad+epsilon*initial_soln-randomization_scale*random_Z)
    cube = subgradient[~active]/lam
    return lam, epsilon, active, betaE, cube, initial_soln

#creating instance X,y,beta: for a single X, sampling lots of y

class instance(object):

    def __init__(self, n, p, s, snr=5, sigma=1., rho=0, random_signs=False, scale =True, center=True):
         (self.n, self.p, self.s,
         self.snr,
         self.sigma,
         self.rho) = (n, p, s,
                     snr,
                     sigma,
                     rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
              np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         self.beta = np.zeros(p)
         self.beta[:self.s] = self.snr
         if random_signs:
             self.beta[:self.s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
         self.active = np.zeros(p, np.bool)
         self.active[:self.s] = True

    def _noise(self):
        return np.random.standard_normal(self.n)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + self._noise()) * self.sigma
        return self.X, Y, self.beta * self.sigma, np.nonzero(self.active)[0], self.sigma

