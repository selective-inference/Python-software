import numpy as np
from sklearn.linear_model import Lasso
from ..constraints.affine import (constraints, selection_interval,
                                 interval_constraints,
                                 sample_from_constraints,
                                 one_parameter_MLE,
                                 gibbs_test,
                                 stack)
from ..distributions.discrete_family import discrete_family

from lasso import lasso, instance

from sklearn.cluster import AgglomerativeClustering

from scipy.stats import norm as ndist, t as tdist

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('cvx not available')
    pass

DEBUG = False

import pdb




    
class stab_lasso(object):

    alpha = 0.05
    UMAU = False

    
    def __init__(self, y, X, lam, n_split = 1, size_split = None, k=None, sigma=1, connectivity=None):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        y : np.float(y)
            The target, in the model $y = X\beta$

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        lam : np.float
            Coefficient of the L-1 penalty in
            $\text{minimize}_{\beta} \frac{1}{2} \|y-X\beta\|^2_2 + 
                \lambda\|\beta\|_1$

        sigma : np.float
            Standard deviation of the gaussian distribution :
            The covariance matrix is
            `sigma**2 * np.identity(X.shape[0])`.
            Defauts to 1.
        """
        self.y = y
        self.X = X
        self.sigma = sigma
        n, p = X.shape
        self.lagrange = lam / n
        self.n_split = n_split
        if size_split == None:
            size_split = n
        self.size_split = size_split
        if k == None:
            k = p
        self._n_clusters = k
        self.connectivity = connectivity

        self._covariance = self.sigma**2 * np.identity(X.shape[0])

    @staticmethod
    def projection(X, k, connectivity):
        """
        Take the data, and returns a matrix, to reduce the dimension
        Returns, P, invP, (P.(X.T)).T
        """
        n, p = X.shape

        
        P = np.identity(p)

        
        ward = AgglomerativeClustering(n_clusters = k, connectivity=connectivity)
        ward.fit(X.T)

        labels = ward.labels_
        P = np.zeros((k, p))
        for i in range(p):
            P[labels[i] ,i] = 1.
        P_inv = np.copy(P)
        P_inv = P_inv.T
        
        s_array = P.sum(axis = 0)
        P = P/s_array
            
        # P_inv = np.linalg.pinv(P)
        
        X_proj = np.dot(P, X.T).T

        
        return P, P_inv, X_proj

    
    def fit(self, sklearn_alpha=None, **lasso_args):
        
        X = self.X
        y = self.y
        n, p = X.shape
        sigma = self.sigma
        lam = self.lagrange * n
        n_split = self.n_split
        size_split = self.size_split
        n_clusters = self._n_clusters
        connectivity = self.connectivity

        # pdb.set_trace()
        cons_list = []
        beta_array = np.zeros((p, n_split))

        for i in range(n_split):
            split = np.random.choice(n, size_split, replace = False)
            # split.sort()        
            
            # X_splitted = X[split,:]
            # y_splitted = y[split]
            X_splitted = X
            y_splitted = y

            #P, P_inv, X_proj = self.projection(X_splitted, n_clusters, connectivity)
            P, P_inv, X_proj = np.identity(p), np.identity(p), X_splitted
            lasso_splitted = lasso(y_splitted, X_proj, lam, sigma)

            lasso_splitted.fit(sklearn_alpha, **lasso_args)

            beta_proj = lasso_splitted.soln
            beta = np.dot(P_inv, beta_proj)
            
            beta_array[:,i] = beta
            constraint_splitted = lasso_splitted.constraints

            # linear_part_splitted = constraint_splitted.linear_part
            # offset = constraint_splitted.offset

            # n_inequality, _ = linear_part_splitted.shape
            # linear_part = np.zeros((n_inequality, n))
            # linear_part[:, split] = linear_part_splitted

            # constraint = constraints(linear_part, offset)
            constraint = constraint_splitted
            cons_list.append(constraint)

        beta = beta_array.mean(axis=1)
        self._constraints = stack(*cons_list)
        self._soln = beta
        self._beta_array = beta_array



    @property
    def soln(self):
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln


    @property
    def constraints(self):
        return self._constraints

    @property
    def intervals(self):
        if hasattr(self, "_intervals"):
            return self._intervals

        X = self.X
        y = self.y
        n, p = X.shape
        n_split = self.n_split

        beta_array = self._beta_array

        C = self._constraints

        eta_array = np.zeros((n_split, n, p))
        
        for k in range(n_split):
            beta = beta_array[:, k]
            active = (beta != 0)
            X_E = X[:, active]
            try:
                XEinv = np.linalg.pinv(X_E)
            except:
                XEinv = None
                pass
            
            if XEinv is not None:
                eta_array[k, :, active] = XEinv

        eta_array = (1./n_split)*eta_array.sum(0)

        self._intervals = []
        for i in range(p):
            eta = eta_array[:, i]

            if eta.any():
                _interval = C.interval(eta, self.y, alpha=self.alpha,
                                       UMAU=self.UMAU)
                self._intervals.append((i, eta, (eta*self.y).sum(), _interval))
        return self._intervals


    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
                       " for selection."):
        if hasattr(self, "_pvals"):
            return self._pvals

        X = self.X
        y = self.y
        n, p = X.shape
        n_split = self.n_split
        

        C = self._constraints

        beta_array = self._beta_array
        eta_array = np.zeros((n_split, n, p))
        
        for k in range(n_split):
            beta = beta_array[:, k]

            if any(beta):
                active = (beta != 0)
                X_E = X[:, active]
                XEinv = np.linalg.pinv(X_E)
            else:
                XEinv = None
            if XEinv is not None:
                eta_array[k, :, active] = XEinv

        eta_array = (1./n_split)*eta_array.sum(0)

        self._pvals = []
        for i in range(p):
            eta = eta_array[:, i]

            if eta.any():
                _pval = C.pivot(eta, self.y)
                _pval = 2 * min(_pval, 1 - _pval)
                                    
                self._pvals.append((i,_pval))
        return self._pvals

    

def test(lam, n_split, size_split):
    A = instance(n = 25, p = 50, s = 5, sigma = 5.)
    X, y, beta, active, sigma = A
    B = stab_lasso(y, X, lam, n_split=n_split, size_split=size_split)
    B.fit()
    # print ("Model fitted")
    I = B.intervals
    P = B.active_pvalues
    return B, P, I


    
def multiple_test(N):
    for i in range(N):
        print i
        test()
    return "bwah"
