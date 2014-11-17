import numpy as np

# rpy2 is used to fit the isotonic regression

import rpy2.robjects as rpy
from rpy2.robjects.numpy2ri import numpy2ri
rpy.conversion.py2ri = numpy2ri

# local imports

from .constraints import (constraints, interval_constraints, selection_interval)
from .chisq import quadratic_test

def _isoreg(y):
    """
    Compute the isotonic fit.
    """
    rpy.r.assign('Y', np.asarray(y))
    return np.array(rpy.r('isoreg(Y)$yf'))

class isotonic(object):

    def __init__(self, y, subgrad_tol=1.e-3,
                 sigma=1.):

        """
        Solve the isotonic regression problem and return the constraints
        implied in the solution.

        Parameters
        ----------

        y : `np.float((n,))`
            Response

        subgrad_tol : float
            Tolerance to determine active constraints.

        Computes
        --------

        fitted : `np.float((n,))`
            Vector of fitted alues.

        constraints : `np.float((n-1,n))`
            Linear constraints implied by the active variables. The number of
            constraints returned depends on the number of nonzero jumps in `mu`.

        subgrad : `np.float((n-1,))`
            Subgradient of nonnegative constraints: it should be elementwise
            nonpositive.

        """

        n = y.shape[0]
        X = np.tril(np.ones((n,n)),0)[:,1:]
        X -= X.mean(0)
        self.X = X
        self.Y = y
        fitted = _isoreg(y)
        subgrad = np.dot(X.T, y - fitted)
        self.active = subgrad > -subgrad_tol    
        self.inactive = ~self.active

        # inefficient, but this is just a prototype
        if self.active.sum():
            self.soln_active = np.linalg.pinv(X[:,self.active])
            self.P_active = np.dot(X[:,self.active], self.soln_active)
            A = np.vstack([-np.dot(X[:,self.inactive].T, 
                                    np.identity(n) - self.P_active),
                            self.soln_active])
        else:
            A = -X.T

        self.fitted, self.subgrad = fitted, subgrad
        self.constraints = constraints((A, np.zeros(A.shape[0])), None,
                                       covariance = sigma**2 * 
                                       np.identity(self.Y.shape[0]))
    
    # Various test statistics we might consider

    @property
    def first_jump(self):
        n = self.Y.shape[0]
        D = (np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)
        jumps = np.nonzero(Dmu)[0]
        if jumps.sum() > 0:
            first_idx = jumps.min()
            eta = D[first_idx]
            return self.constraints.pivots(eta, self.Y)
        else:
            return None

    @property
    def largest_jump(self):
        n = self.Y.shape[0]
        D = -(np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)
        jumps = np.nonzero(Dmu)[0]
        if jumps.sum() > 0:
            max_idx = np.argmax(Dmu)
            self.fit_matrix = self.P_active + np.ones((n,n), np.float) / n
            eta = np.dot(D[max_idx], self.fit_matrix)

            diffD = D - D[max_idx]
            indices = range(diffD.shape[0])
            indices.pop(max_idx)
            diffD = np.dot(diffD[indices], self.fit_matrix)

            all_constraints = np.vstack([-diffD, self.constraints.inequality])
            return interval_constraints(all_constraints, \
                             np.zeros(all_constraints.shape[0]), 
                             self.constraints.covariance,
                             self.Y,
                             eta)
        else:
            return None

    @property
    def largest_jump_univariate(self):
        n = self.Y.shape[0]
        D = -(np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)
        jumps = np.nonzero(Dmu)[0]
        if jumps.sum() > 0:
            max_idx = np.argmax(Dmu)
            self.fit_matrix = self.P_active + np.ones((n,n), np.float) / n
            eta = np.zeros(n)
            eta[:max_idx] = 1
            eta -= eta.mean()
            eta /= np.linalg.norm(eta)

            diffD = D - D[max_idx]
            indices = range(diffD.shape[0])
            indices.pop(max_idx)
            diffD = np.dot(diffD[indices], self.fit_matrix)

            all_constraints = np.vstack([-diffD, self.constraints.inequality])
            return interval_constraints(all_constraints, \
                             np.zeros(all_constraints.shape[0]), 
                             self.constraints.covariance,
                             self.Y,
                             eta,
                             two_sided=True)
        else:
            return None

    @property
    def largest_jump_interval(self):
        n = self.Y.shape[0]
        D = -(np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)
        jumps = np.nonzero(Dmu)[0]
        if jumps.sum() > 0:
            max_idx = np.argmax(Dmu)
            self.fit_matrix = self.P_active + np.ones((n,n), np.float) / n
            eta = np.dot(D[max_idx], self.fit_matrix)

            diffD = D - D[max_idx]
            indices = range(diffD.shape[0])
            indices.pop(max_idx)
            diffD = np.dot(diffD[indices], self.fit_matrix)

            all_constraints = np.vstack([-diffD, self.constraints.inequality])
            return eta, selection_interval(all_constraints, \
                             np.zeros(all_constraints.shape[0]), 
                             self.constraints.covariance,
                             self.Y,
                             eta, dps=22,
                             upper_target=0.95,
                             lower_target=0.05)
        else:
            return None

    @property
    def largest_jump_univariate_interval(self):
        n = self.Y.shape[0]
        D = -(np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)
        jumps = np.nonzero(Dmu)[0]
        if jumps.sum() > 0:
            max_idx = np.argmax(Dmu)
            self.fit_matrix = self.P_active + np.ones((n,n), np.float) / n
            eta = np.zeros(n)
            eta[:max_idx] = 1
            eta -= eta.mean()
            eta /= np.linalg.norm(eta)

            diffD = D - D[max_idx]
            indices = range(diffD.shape[0])
            indices.pop(max_idx)
            diffD = np.dot(diffD[indices], self.fit_matrix)

            all_constraints = np.vstack([-diffD, self.constraints.inequality])
            return eta, selection_interval(all_constraints, \
                             np.zeros(all_constraints.shape[0]), 
                             self.constraints.covariance,
                             self.Y,
                             eta, dps=22,
                             upper_target=0.95,
                             lower_target=0.05)
        else:
            return None

    def sum_jumps(self, idx):
        n = self.Y.shape[0]
        D = -(np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)

        jumps = np.fabs(Dmu) > 1.0e-3 * np.fabs(Dmu).max()
        if jumps.sum() > 0:

            idx = min(idx, jumps.sum())
            order_idx = np.argsort(Dmu)[::-1]
            orderedD = D[order_idx][:jumps.sum()]

            self.fit_matrix = self.P_active + np.ones((n,n), np.float) / n
            orderedD = np.dot(orderedD, self.fit_matrix)

            if idx < jumps.sum():
                A = np.zeros((orderedD.shape[0], orderedD.shape[0]))

                for i in range(idx):
                    A[i,i] = 1
                    A[i,idx] = -1
                for j in range(idx, A.shape[0]-1):
                    A[j,j] = 1
                    A[j,j+1] = -1
                A[-1,-1] = 1
            else:
                A = np.identity(idx)

            all_constraints = np.vstack([np.dot(A, orderedD), self.constraints.inequality])

            con = constraints((all_constraints, 
                               np.zeros(all_constraints.shape[0])), None,
                              covariance = self.constraints.covariance)
            eta = orderedD[:idx].sum(0)
            return con.pivots(eta, self.Y)
        else:
            return None
        
    def sum_jumps_interval(self, idx):
        n = self.Y.shape[0]
        D = -(np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)

        jumps = np.fabs(Dmu) > 1.0e-3 * np.fabs(Dmu).max()
        if jumps.sum() > 0:

            idx = min(idx, jumps.sum())
            order_idx = np.argsort(Dmu)[::-1]
            orderedD = D[order_idx][:jumps.sum()]

            self.fit_matrix = self.P_active + np.ones((n,n), np.float) / n
            orderedD = np.dot(orderedD, self.fit_matrix)

            if idx < jumps.sum():
                A = np.zeros((orderedD.shape[0], orderedD.shape[0]))

                for i in range(idx):
                    A[i,i] = 1
                    A[i,idx] = -1
                for j in range(idx, A.shape[0]-1):
                    A[j,j] = 1
                    A[j,j+1] = -1
                A[-1,-1] = 1
            else:
                A = np.identity(idx)

            all_constraints = np.vstack([np.dot(A, orderedD), self.constraints.inequality])

            con = constraints((all_constraints, 
                               np.zeros(all_constraints.shape[0])), None,
                              covariance = self.constraints.covariance)
            eta = orderedD[:idx].sum(0)
            return eta, selection_interval(all_constraints, \
                             np.zeros(all_constraints.shape[0]), 
                             self.constraints.covariance,
                             self.Y,
                             eta, dps=22,
                             upper_target=0.95,
                             lower_target=0.05)

        else:
            return None
        
    def combine_jumps(self, idx):
        n = self.Y.shape[0]
        D = -(np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
        Dmu = np.dot(D, self.fitted)
        jumps = np.fabs(Dmu) > 1.0e-3 * np.fabs(Dmu).max()

        if jumps.sum() > 0:

            order_idx = np.argsort(Dmu)[::-1]
            orderedD = D[order_idx]

            self.fit_matrix = self.P_active + np.ones((n,n), np.float) / n
            idx = min(idx, jumps.sum())
            A = np.dot(orderedD[:idx], self.fit_matrix)
            P_test = np.linalg.svd(A, full_matrices=0)[2] 

            diffD = -np.diff(orderedD[:jumps.sum()], axis=0)
            diffD = np.dot(diffD, self.fit_matrix)

            all_constraints = np.vstack([diffD, self.constraints.inequality])

            con = constraints((all_constraints, 
                               np.zeros(all_constraints.shape[0])), None,
                              covariance = self.constraints.covariance)

            return quadratic_test(self.Y, P_test, con)
        else:
            return None
