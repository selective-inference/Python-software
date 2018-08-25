import os, glob, tempfile
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

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(knockoff)')

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

class knockoffs_sigma(generic_method):

    factor_method = 'equi'
    method_name = Unicode('Knockoffs')
    knockoff_method = Unicode("ModelX (asdp)")
    model_target = Unicode("full")
    selectiveR_method = True
    forward_step = False

    @classmethod
    def setup(cls, feature_cov):

        cls.feature_cov = feature_cov
        numpy2ri.activate()

        # see if we've factored this before

        have_factorization = False
        if not os.path.exists('.knockoff_factorizations'):
            os.mkdir('.knockoff_factorizations')
        factors = glob.glob('.knockoff_factorizations/*npz')
        for factor_file in factors:
            factor = np.load(factor_file)
            feature_cov_f = factor['feature_cov']
            if ((feature_cov_f.shape == feature_cov.shape) and
                (factor['method'] == cls.factor_method) and
                np.allclose(feature_cov_f, feature_cov)):
                have_factorization = True
                cls.knockoff_chol = factor['knockoff_chol']

        if not have_factorization:
            print('doing factorization')
            cls.knockoff_chol = factor_knockoffs(feature_cov, cls.factor_method)

        numpy2ri.deactivate()

    def select(self):

        numpy2ri.activate()
        rpy.r.assign('chol_k', self.knockoff_chol)
        rpy.r('''
        knockoffs = function(X) {
           mu = rep(0, ncol(X))
           mu_k = X # sweep(X, 2, mu, "-") %*% SigmaInv_s
           X_k = mu_k + matrix(rnorm(ncol(X) * nrow(X)), nrow(X)) %*% 
            chol_k
           return(X_k)
        }
            ''')
        numpy2ri.deactivate()

        try:
            numpy2ri.activate()
            rpy.r.assign('X', self.X)
            rpy.r.assign('Y', self.Y)
            rpy.r.assign('q', self.q)
            if self.forward_step:
                rpy.r('V=knockoff.filter(X, Y, fdr=q, knockoffs=knockoffs, stat=stat.forward_selection)$selected')
            else:
                rpy.r('V=knockoff.filter(X, Y, fdr=q, knockoffs=knockoffs)$selected')
            rpy.r('if (length(V) > 0) {V = V-1}')
            V = rpy.r('V')
            numpy2ri.deactivate()
            return np.asarray(V, np.int), np.asarray(V, np.int)
        except:
            return [], []

knockoffs_sigma.register()

def factor_knockoffs(feature_cov, method='asdp'):

    numpy2ri.activate()
    print(feature_cov, 'blah')
    rpy.r.assign('Sigma', feature_cov)
    rpy.r.assign('method', method)
    rpy.r('''

    # Compute the Cholesky -- from create.gaussian

    Sigma = as.matrix(Sigma)
    print(dim(Sigma))
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
