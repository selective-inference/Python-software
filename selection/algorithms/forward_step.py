"""
In this module, we implement forward stepwise model selection for $K$ steps.

The main goal of this is to produce a set of linear inequality constraints satisfied by
$y$ after $K$ steps.

"""

import warnings
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

# local imports 

from ..constraints.affine import constraints, gibbs_test, stack
from ..distributions.chisq import quadratic_test
from ..distributions.discrete_family import discrete_family
from .projection import projection

DEBUG = False

class forward_step(object):

    """
    Centers columns of X!
    """

    def __init__(self, X, Y, 
                 subset=[],
                 covariance=None):
        self.subset = subset
        self.X, self.Y = X, Y
        if subset != []:
            self.adjusted_X = X.copy()[subset]
            self.subset_subset = Y.copy()[subset]
            self.subset_selector = np.identity(self.X.shape[0])[subset]
        else:
            self.adjusted_X = X.copy()
            self.subset_Y = Y.copy()

        self.P = [None] # residual forming projections
        self.variables = []
        self.Z = []
        self.Zfunc = []
        self.signs = []
        self.covariance = covariance
        self._resid_vector = self.subset_Y.copy() 

    def __iter__(self):
        n, p = self.X.shape
        self.identity_cone = []
        self.inactive = range(p)
        self.offset = [[np.ones(p) * np.inf, np.ones(p) * np.inf]]
        return self

    def next(self, compute_pval=False,
             use_identity=False,
             burnin=2000,
             ndraw=8000):
        """
        """
        
        adjusted_X, Y, inactive = self.adjusted_X, self.subset_Y, self.inactive
        resid_vector = self._resid_vector
        n, p = adjusted_X.shape

        # up to now inactive
        inactive = sorted(set(range(p)).difference(self.variables))
        scale = np.sqrt(np.sum(adjusted_X**2, 0))
        
        linear_part = []

        Zstat = np.dot(adjusted_X.T[inactive], Y) / scale[inactive]
        idx = np.argmax(np.fabs(Zstat))
        next_var = inactive[idx]
        next_sign = np.sign(Zstat[idx])

        realized_Z = Zstat[idx]

        # keep track of identity for testing
        # variables other than the last one added

        keep = np.zeros(p, np.bool)
        keep[inactive] = True
        keep[next_var] = False
        identity_linpart = np.vstack([adjusted_X[:,keep].T 
                                      / scale[keep,None] -
                                      next_sign * adjusted_X[:,next_var] / scale[next_var],
                                      -adjusted_X[:,keep].T 
                                      / scale[keep,None] -
                                      next_sign * adjusted_X[:,next_var] / scale[next_var],
                                      -next_sign * adjusted_X[:,next_var].reshape((1,-1))])

        identity_con = constraints(identity_linpart,
                                   np.zeros(identity_linpart.shape[0]))

        self.identity_cone.append(identity_linpart)

        eta = adjusted_X[:,next_var]

        if compute_pval:

            XI = self.X[:,inactive]
            linear_part = np.vstack([XI.T, -XI.T])
            offset = np.array(self.offset)
            offset = offset[:,:,inactive]
            offset_pos = np.min(offset[:,0], 0)
            offset_neg = np.min(offset[:,1], 0)
            offset = np.hstack([offset_pos, offset_neg])
            con = constraints(linear_part, offset,
                              covariance=self.covariance)

            if use_identity:
                con = stack(con, identity_con)
                con.covariance = self.covariance
            if self.variables:
                XA = self.X[:,self.variables]
                self.sequential_con = con.conditional(XA.T,
                                                      np.dot(XA.T, Y))
            else:
                self.sequential_con = con

            def maxT(Z, L=adjusted_X[:,inactive], S=scale[inactive]):
                Tstat = np.fabs(np.dot(Z, L) / S[None,:]).max(1)
                return Tstat

            pval = gibbs_test(self.sequential_con,
                              Y,
                              eta,
                              sigma_known=True,
                              white=False,
                              ndraw=ndraw,
                              burnin=burnin,
                              how_often=-1,
                              UMPU=False,
                              use_random_directions=False,
                              tilt=None,
                              alternative='greater',
                              test_statistic=maxT,
                              )[0]

        # now update state for next step

        inactive.pop(idx)
        self.variables.append(next_var); self.signs.append(next_sign)

        realized_Z_adjusted = np.fabs(realized_Z) * scale
        offset_shift = np.dot(self.X.T, Y - resid_vector)
        self.offset.append([realized_Z_adjusted + offset_shift,
                            realized_Z_adjusted - offset_shift])

        resid_vector -= realized_Z * adjusted_X[:,next_var] / scale[next_var]
        adjusted_X -= (np.multiply.outer(eta, 
                                         np.dot(eta,
                                                adjusted_X)) / 
                       (eta**2).sum())
        if compute_pval:
            return pval

    def constraints(self, step=np.inf, identify_last_variable=True):
        default_step = len(self.variables)
        if default_step > 0 and not identify_last_variable:
            default_step -= 1
        step = min(step, default_step)
        A = np.vstack(self.identity_cone[:step])

        if self.subset == []:
            return constraints(A, 
                               np.zeros(A.shape[0]), 
                               covariance=self.covariance)
        return constraints(np.dot(A, 
                                  self.subset_selector),
                           np.zeros(A.shape[0]), 
                           covariance=self.covariance)

    def model_pivots(self, which_step, alternative='greater',
                     saturated=True,
                     ndraw=5000,
                     burnin=2000,
                     which_var=[], 
                     compute_intervals=False,
                     nominal=False,
                     coverage=0.95):
        """
        Compute two-sided pvalues for each coefficient
        in a given step of forward stepwise.

        Parameters
        ----------

        which_step : int
            Which step of forward stepwise.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        saturated : bool
            Use saturated model or selected model?

        ndraw : int (optional)
            Defaults to 5000.

        burnin : int (optional)
            Defaults to 2000.

        which_var : []
            Compute pivots for which variables? If empty,
            return a pivot for all selected variable at stage `which_step`.

        compute_intervals : bool
            Should we compute intervals?

        coverage : float
            Coverage for intervals, if computed.

        Returns
        -------

        pivots : list
             List of (variable, pvalue)
             for selected model.

        """

        if which_step == 0:
            return []

        if self.covariance is None and saturated:
            raise ValueError('need a covariance matrix to compute pivots for saturated model')

        con = copy(self.constraints())

        if self.covariance is not None:
            con.covariance[:] = self.covariance 

        linear_part = self.X[:,self.variables[:which_step]]
        observed = np.dot(linear_part.T, self.Y)
        LSfunc = np.linalg.pinv(linear_part)

        if which_var == []:
            which_var = self.variables[:which_step]

        pivots = []

        if compute_intervals:
            if self.covariance is None:
                raise ValueError('covariance must be known for computing intervals')
            intervals = []

        if saturated:
            for i in range(LSfunc.shape[0]):
                if self.variables[i] in which_var:
                    pivots.append((self.variables[i],
                                   con.pivot(LSfunc[i], self.Y,
                                             alternative=alternative)))
                  
        else:
            sigma_known = self.covariance is not None
            for i in range(LSfunc.shape[0]):
                if self.variables[i] in which_var:
                    keep = np.ones(LSfunc.shape[0], np.bool)
                    keep[i] = False

                    if which_step > 1:
                        conditional_law = con.conditional(linear_part.T[keep],
                                                          observed[keep])
                    else:
                        conditional_law = con

                    eta = LSfunc[i] * self.signs[i]
                    observed_func = (eta*self.Y).sum()
                    if compute_intervals:
                        _, _, _, family = gibbs_test(conditional_law,
                                                     self.Y,
                                                     eta,
                                                     sigma_known=True,
                                                     white=False,
                                                     ndraw=ndraw,
                                                     burnin=burnin,
                                                     how_often=10,
                                                     UMPU=False,
                                                     use_random_directions=False,
                                                     tilt=np.dot(conditional_law.covariance, 
                                                                 eta))

                        lower_lim, upper_lim = family.equal_tailed_interval(observed_func, 1 - coverage)

                        # in the model we've chosen, the parameter beta is associated
                        # to the natural parameter as below
                        # exercise: justify this!

                        lower_lim_final = np.dot(eta, np.dot(conditional_law.covariance, eta)) * lower_lim
                        upper_lim_final = np.dot(eta, np.dot(conditional_law.covariance, eta)) * upper_lim

                        intervals.append((self.variables[i], (lower_lim_final, upper_lim_final)))
                    else: # we do not really need to tilt just for p-values
                        _ , _, _, family = gibbs_test(conditional_law,
                                                      self.Y,
                                                      eta,
                                                      sigma_known=True,
                                                      white=False,
                                                      ndraw=ndraw,
                                                      burnin=burnin,
                                                      how_often=10,
                                                      use_random_directions=False,                                                     UMPU=False,
                                                      alternative=alternative)

                    pval = family.cdf(0, observed_func)
                    if alternative == 'twosided':
                        pval = 2 * min(pval, 1 - pval)
                    elif alternative == 'greater':
                        pval = 1 - pval
                    pivots.append((self.variables[i], 
                                   pval))

        return pivots

    def model_quadratic(self, which_step):
        LSfunc = np.linalg.pinv(self.X[:,self.variables[:which_step]])
        P_LS = np.linalg.svd(LSfunc, full_matrices=False)[2]
        return quadratic_test(self.Y, P_LS, self.constraints)

def info_crit_stop(Y, X, sigma, cost=2,
                   subset=[]):
    """
    Fit model using forward stepwise,
    stopping using a rule like AIC or BIC.
    
    The error variance must be supplied, in which
    case AIC is essentially Mallow's C_p.

    Parameters
    ----------

    Y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : float (optional)
        Error variance.

    cost : float
        Cost per parameter. For BIC use cost=log(X.shape[0])

    subset : []
        Subset of cases to use for selection, defaults to [].

    Returns
    -------

    FS : `forward_stepwise`
        Instance of forward stepwise stopped at the
        corresponding step. Constraints of FS
        will reflect the minimum Z score requirement.

    """
    n, p = X.shape
    FS = forward_stepwise(X, Y, covariance=sigma**2 * np.identity(n), subset=subset)
    while True:
        FS.next()

        if FS.Z[-1] < sigma * np.sqrt(cost):
            break

    new_linear_part = -np.array(FS.Zfunc)
    new_linear_part[-1] *= -1
    new_offset = -sigma * np.sqrt(cost) * np.ones(new_linear_part.shape[0])
    new_offset[-1] *= -1

    if subset != []:
        new_con = stack(FS.constraints, constraints(np.dot(new_linear_part, FS.subset_selector),
                                                    new_offset))
    else:
        new_con = stack(FS.constraints, constraints(new_linear_part,new_offset))
    new_con.covariance[:] = sigma**2 * np.identity(n)
    if DEBUG:
        print FS.constraints.linear_part.shape, 'before'
    FS._constraints = new_con
    if DEBUG:
        print FS.constraints.linear_part.shape, 'should have added number of steps constraints'
    FS.active = FS.variables[:-1]
    return FS

def data_carving_IC(y, X, sigma,
                    cost=2.,
                    stage_one=None,
                    split_frac=0.9,
                    coverage=0.95, 
                    ndraw=8000,
                    burnin=2000,
                    saturated=False,
                    splitting=False,
                    compute_intervals=True):

    """
    Fit a LASSO with a default choice of Lagrange parameter
    equal to `lam_frac` times $\sigma \cdot E(|X^T\epsilon|)$
    with $\epsilon$ IID N(0,1) on a proportion (`split_frac`) of
    the data.

    Parameters
    ----------

    y : np.float
        Response vector

    X : np.float
        Design matrix

    sigma : np.float
        Noise variance

    stage_one : [np.array(np.int), None] (optional)
        Index of data points to be used in  first stage.
        If None, a randomly chosen set of entries is used based on
        `split_frac`.

    split_frac : float (optional)
        What proportion of the data to use in the first stage?
        Defaults to 0.9.

    coverage : float
        Coverage for selective intervals. Defaults to 0.95.

    ndraw : int (optional)
        How many draws to keep from Gibbs hit-and-run sampler.
        Defaults to 8000.

    burnin : int (optional)
        Defaults to 2000.

    splitting : bool (optional)
        If True, also return splitting pvalues and intervals.
      
    Returns
    -------

    results : [(variable, pvalue, interval)
        Indices of active variables, 
        selected (twosided) pvalue and selective interval.
        If splitting, then each entry also includes
        a (split_pvalue, split_interval) using stage_two
        for inference.

    """

    n, p = X.shape
    if stage_one is None:
        splitn = int(n*split_frac)
        indices = np.arange(n)
        np.random.shuffle(indices)
        stage_one = indices[:splitn]
        stage_two = indices[splitn:]
    else:
        stage_two = [i for i in np.arange(n) if i not in stage_one]
    y1, X1 = y[stage_one], X[stage_one]
    splitn = len(stage_one)

    FS = info_crit_stop(y, X, sigma, cost=cost, subset=stage_one)
    active = FS.active
    s = len(active)

    LSfunc = np.linalg.pinv(FS.X[:,active])

    if splitn < n and splitting:

        y2, X2 = y[stage_two], X[stage_two]
        X_E2 = X2[:,active]
        X_Ei2 = np.linalg.pinv(X_E2)
        beta_E2 = np.dot(X_Ei2, y2)
        inv_info_E2 = np.dot(X_Ei2, X_Ei2.T)

        splitting_pvalues = []
        splitting_intervals = []

        split_cutoff = np.fabs(ndist.ppf((1. - coverage) / 2))

        if n - splitn < s:
            warnings.warn('not enough data for second stage of sample splitting')

        for j in range(LSfunc.shape[0]):
            if s < n - splitn: # enough data to generically
                               # test hypotheses. proceed as usual

                split_pval = ndist.cdf(beta_E2[j] / (np.sqrt(inv_info_E2[j,j]) * sigma))
                split_pval = 2 * min(split_pval, 1. - split_pval)
                splitting_pvalues.append(split_pval)

                splitting_interval = (beta_E2[j] - 
                                      split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma,
                                      beta_E2[j] + 
                                      split_cutoff * np.sqrt(inv_info_E2[j,j]) * sigma)
                splitting_intervals.append(splitting_interval)
            else:
                splitting_pvalues.append(np.random.sample())
                splitting_intervals.append((np.nan, np.nan))
    elif splitting:
        splitting_pvalues = np.random.sample(LSfunc.shape[0])
        splitting_intervals = [(np.nan, np.nan)] * LSfunc.shape[0]

    result = FS.model_pivots(len(active),
                             saturated=saturated,
                             ndraw=ndraw,
                             burnin=burnin,
                             compute_intervals=compute_intervals)

    if compute_intervals:
        pvalues, intervals = result
    else:
        pvalues = result
        intervals = [(v, (np.nan, np.nan)) for v in active]

    pvalues = [p for _, p in pvalues]
    intervals = [interval for _, interval in intervals]

    if not splitting:
        return zip(active, 
                   pvalues,
                   intervals), FS
    else:
        return zip(active, 
                   pvalues,
                   intervals,
                   splitting_pvalues,
                   splitting_intervals), FS

