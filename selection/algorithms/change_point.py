from copy import copy

import numpy as np, pandas as pd

import regreg.api as rr
import regreg.affine as ra
from regreg.affine.multiscale import multiscale, choose_tuning_parameter

from .sqrt_lasso import solve_sqrt_lasso
from .lasso import lasso
from ..constraints.affine import constraints

class change_point(object):

    """                                                                         
                                                                                
    Estimate change points using multiscale changepoint                         
    penatly.                                                                    
                                                                                
    """

    # parameters for choosing tuning parameter

    ndraw = 50
    quantile = 0.95
    sigma = None

    def __init__(self, Y, minsize=None):

        self.Y = Y
        self.p = p = Y.shape[0]
        self.M = multiscale(p)
        self.M.scaling = np.sqrt(self.M.sizes)
        self.tuning_param = choose_tuning_parameter(self.M, 
                                           ndraw=self.ndraw, 
                                           quantile=self.quantile, 
                                           sigma=self.sigma)
        self.feature_weights = (self.tuning_param + np.sqrt(2 * np.log(p / self.M.sizes))) / np.sqrt(p)

    def fit(self, solve_args={'min_its':200}):

        Y, M, p, weights = self.Y, self.M, self.Y.shape[0], self.feature_weights

        Y0 = Y - Y.mean()
        coef = solve_sqrt_lasso(M.T, Y0, weights=weights, solve_args=solve_args)[0]

        self.active = active = coef != 0
        self.inactive = inactive = ~active

        if active.sum():
            jumps = self.merge_intervals(M.slices[active])
            M_inactive = copy(M)
            M_inactive.update_slices(M.slices[inactive])
            X = M.form_matrix(M.slices[active])[0]
            R = np.identity(p) - X.dot(np.linalg.pinv(X))
            L_inactive = M_inactive.dot(rr.astransform(R))
            L = lasso.sqrt_lasso(X, Y0, weights[active])
            L.fit(solve_args={'min_its':200}, lasso_solution=coef[active])
            C = L.constraints.linear_part.dot(X.T)
            L_inactive_neg = ra.scalar_multiply(L_inactive, -1.)

            irrep = L_inactive.dot(X.dot(-np.diag(L.active_signs).dot(L.constraints.offset)))
            full_lin = ra.vstack([L_inactive, L_inactive_neg, C])
            full_offset = np.hstack([irrep + L._multiplier * weights[inactive],
                                     -irrep + L._multiplier * weights[inactive], 
                                     L.constraints.offset])

            full_con = constraints(full_lin, full_offset, covariance=L._sigma_hat**2 * np.identity(p))

            fit = X.dot(coef[active]) + Y.mean()

            segments = np.array([jumps[:-1],jumps[1:]]).T
            Xr = M.form_matrix(segments)[0]
            relaxed_fit = Xr.dot(np.linalg.pinv(Xr).dot(Y)) + Y.mean()
#             S = L.summary('onesided')
#             pS = pd.DataFrame(S)
#             pS['interval'] = [M.slices[i] for i in np.nonzero(active)[0]]
#             pS = pS.reindex(columns=['interval', 'pval', 'lasso', 'onestep'])
            pS = None
        else:
            fit = relaxed_fit = Y.mean() * np.ones(self.p)
            pS = None
            segments = np.array([[0,self.p]])
        return fit, relaxed_fit, pS, segments

    def merge_intervals(self, intervals):
        """
        cluster the endpoints into intervals
        by saying points that are M.minsize apart or less
        are in the same cluster
        """

        M = self.M

        vertices = sorted(list(np.unique(np.hstack([intervals['start'], intervals['end']]))) + [0, M.input_shape[0]-1])
        k = len(vertices)
        clusters = []
        cur_cluster = [vertices[0]]
        for j in range(1, k):
            if vertices[j] - vertices[j-1] < M.minsize:
                cur_cluster.append(vertices[j])
            else:
                clusters.append(cur_cluster)
                cur_cluster = [vertices[j]]
        clusters.append(cur_cluster)

        j = np.array([int(np.median(c)) for c in clusters])
        if j[0] < M.minsize: 
            j[0] = 0
        if j[-1] > M.input_shape[0] - 1 - M.minsize:
            j[-1] = M.input_shape[0] 
        return j

def one_jump_instance(delta, p=60, sigma=1):
    """
    Data generating mechanism of Figure 1 in [http://arxiv.org/abs/1606.03552](http://arxiv.org/abs/1606.03552).

    Parameters
    ----------

    delta : float
        Signal size

    p : int
        Shape of signal.

    sigma : float
        Noise variance -- both signal and noise are scaled by this scalar.

    """
    signal = np.zeros(p)
    signal[(p/2):] += delta * sigma
    y = np.random.standard_normal(p) * sigma + signal
    return y, signal

