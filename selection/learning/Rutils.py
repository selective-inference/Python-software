import os, glob, tempfile, warnings
import numpy as np

from traitlets import (HasTraits, 
                       Integer, 
                       Unicode, 
                       Float, 
                       Integer, 
                       Instance, 
                       Dict, 
                       Bool,
                       default)
# Rpy

try:
    import rpy2.robjects as rpy
    from rpy2.robjects import numpy2ri
    rpy.r('library(knockoff); library(glmnet)')
    from rpy2 import rinterface
except ImportError:
    warnings.warn("rpy2 with knockoff and glmnet unavailable")

def null_print(x):
    pass

# Knockoff selection

methods = {}
class generic_method(HasTraits):

    need_CV = False
    selectiveR_method = False
    wide_ok = True # ok for p>= n?

    # Traits

    q = Float(0.2)
    method_name = Unicode('Generic method')
    model_target = Unicode()

    @classmethod
    def setup(cls, feature_cov):
        cls.feature_cov = feature_cov

    def __init__(self, X, Y, l_theory, l_min, l_1se, sigma_reid):
        (self.X,
         self.Y,
         self.l_theory,
         self.l_min,
         self.l_1se,
         self.sigma_reid) = (X,
                             Y,
                             l_theory,
                             l_min,
                             l_1se,
                             sigma_reid)

    def select(self):
        raise NotImplementedError('abstract method')

    @classmethod
    def register(cls):
        methods[cls.__name__] = cls

    def selected_target(self, active, beta):
        C = self.feature_cov[active]
        Q = C[:,active]
        return np.linalg.inv(Q).dot(C.dot(beta))

    def full_target(self, active, beta):
        return beta[active]

    def get_target(self, active, beta):
        if self.model_target not in ['selected', 'full', 'debiased']:
            raise ValueError('Gaussian methods only have selected or full targets')
        if self.model_target in ['full', 'debiased']:
            return self.full_target(active, beta)
        else:
            return self.selected_target(active, beta)

class lasso_glmnet(generic_method):

    def select(self, CV=True, seed=0):

        numpy2ri.activate()

        rpy.r.assign('X', self.X.copy())
        rpy.r.assign('Y', self.Y.copy())
        rpy.r('X = as.matrix(X)')
        rpy.r('Y = as.numeric(Y)')
        rpy.r('set.seed(%d)' % seed)
        rpy.r('cvG = cv.glmnet(X, Y, intercept=FALSE, standardize=FALSE)')
        rpy.r("L1 = cvG[['lambda.min']]")
        rpy.r("L2 = cvG[['lambda.1se']]")
        if CV:
            rpy.r("L = L1")
        else:
            rpy.r("L = 0.99 * L2")
        rpy.r("G = glmnet(X, Y, intercept=FALSE, standardize=FALSE)")
        n, p = self.X.shape
        L = rpy.r('L')
        rpy.r('B = as.numeric(coef(G, s=L, exact=TRUE, x=X, y=Y))[-1]')
        B = np.asarray(rpy.r('B'))
        selected = (B != 0)
        if selected.sum():
            V = np.nonzero(selected)[0]
            return V, V
        else:
            return [], []

lasso_glmnet.register()

def factor_knockoffs(feature_cov, method='asdp'):

    numpy2ri.activate()
    rpy.r.assign('Sigma', feature_cov)
    rpy.r.assign('method', method)
    rpy.r('''

    # Compute the Cholesky -- from create.gaussian

    Sigma = as.matrix(Sigma)
    diag_s = diag(switch(method, equi = create.solve_equi(Sigma), 
                  sdp = create.solve_sdp(Sigma), asdp = create.solve_asdp(Sigma)))
    if (is.null(dim(diag_s))) {
        diag_s = diag(diag_s, length(diag_s))
    }
    SigmaInv_s = solve(Sigma, diag_s)
    Sigma_k = 2 * diag_s - diag_s %*% SigmaInv_s
    chol_k = chol(Sigma_k)
    ''')
    knockoff_chol = np.asarray(rpy.r('chol_k'))
    SigmaInv_s = np.asarray(rpy.r('SigmaInv_s'))
    diag_s = np.asarray(rpy.r('diag_s'))
    np.savez('.knockoff_factorizations/%s.npz' % (os.path.split(tempfile.mkstemp()[1])[1],),
             method=method,
             feature_cov=feature_cov,
             knockoff_chol=knockoff_chol)

    return knockoff_chol

def cv_glmnet_lam(X, Y, seed=0):
    """

    Some calculations that can be reused by methods:
    
    lambda.min, lambda.1se, lambda.theory and Reid et al. estimate of noise

    """
    numpy2ri.activate()
    rpy.r('set.seed(%d)' % seed)
    rpy.r.assign('X', X.copy())
    rpy.r.assign('Y', Y.copy())
    rpy.r('X=as.matrix(X)')
    rpy.r('Y=as.numeric(Y)')
    rpy.r('set.seed(1)')
    rpy.r('G = cv.glmnet(X, Y, intercept=FALSE, standardize=FALSE)')
    rpy.r("L = G[['lambda.min']]")
    rpy.r("L1 = G[['lambda.1se']]")
    L = rpy.r('L')
    L1 = rpy.r('L1')
    numpy2ri.deactivate()
    return float(1.00001 * L[0]), float(1.00001 * L1[0]),
