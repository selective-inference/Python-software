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

from ..constraints.affine import constraints, gibbs_test, stack, MLE_family
from ..distributions.chisq import quadratic_test
from .projection import projection

DEBUG = False

class forward_stepwise(object):

    """
    Centers columns of X!
    """

    def __init__(self, X, Y, 
                 subset=[],
                 covariance=None):
        self.subset = subset
        self.X, self.Y = X, Y
        if subset != []:
            self.Xsub = X.copy()[subset]
            self.Xsub -= self.Xsub.mean(0)[None,:]
            self.Ysub = Y.copy()[subset]
            self.Ysub -= self.Ysub.mean()
            self.subset_selector = np.identity(self.X.shape[0])[subset]
        else:
            self.Xsub = X.copy()
            self.Ysub = Y.copy()
        self.P = [None] # residual forming projections
        self.A = None
        self.variables = []
        self.Z = []
        self.Zfunc = []
        self.signs = []
        self.covariance = covariance

    def __iter__(self):
        return self

    def next(self):
        """
        Take one step of forward stepwise.
        Internally, this has the effect of: 

        * adding a new (lowrank) projection to `self.P`, 
        
        * adding a new variable to `self.variables`

        * adding a certain number of rows to `self.A`

        * signs are also tracked (unnecessarily for the moment) in `self.signs`

        The multiplication `np.dot(self.A, eta)` can be made more 
        efficient because the projections are just a list of 
        Gram-Schmidt orthogonalized vectors.

        """
        P = self.P[-1]
        
        X, Y = self.Xsub, self.Ysub
        n, p = self.Xsub.shape

        if P is None: # first step
            U = np.dot(X.T, Y)
            scale = np.sqrt((X**2).sum(0))
            Z = np.fabs(U) / scale
            idx = np.argmax(Z)
            sign = np.sign(U[idx])
            Unew = X[:,idx] / scale[idx]
            Pnew = projection(Unew.reshape((-1,1)))
            self.As = [canonicalA(X, Y, idx, sign, scale=scale)]
            self.A = self.As[0]
            self.variables.append(idx)
            self.signs.append(sign)
            self.Z.append(Z[idx])
            self.Zfunc.append(Unew * sign)
        else:
            RY = Y-P(Y)
            RX = X-P(X)
            keep = np.ones(p, np.bool)
            keep[self.variables] = 0
            RX = RX[:,keep]

            scale = np.sqrt((RX**2).sum(0))
            U = np.dot(RX.T, RY)
            Z = np.fabs(U) / scale
            idx = np.argmax(Z)

            sign = np.sign(U[idx])
            self.variables.append(np.arange(p)[keep][idx])
            self.signs.append(sign)
            self.Z.append(Z[idx])
            self.Zfunc.append((RX.T[idx] / scale[idx]) * sign)
            Unew = RX[:,idx] / scale[idx]
            Pnew = P.stack(Unew.reshape((-1,1)))
            newA = canonicalA(RX, RY, idx, sign, scale=scale)
            self.As.append(newA)
            if DEBUG:
                print np.linalg.norm(np.dot(newA, Y) - np.dot(newA, RY)), 'should be 0'
                print np.linalg.norm(P(newA.T)), np.linalg.norm(P(RX)), 'newA'
            self.A = np.vstack([self.A, newA])

        if DEBUG:
            Pother = np.linalg.svd(X[:,self.variables], full_matrices=0)[0]
            print np.linalg.norm(Pother - Pnew(Pother)), 'Pnorm'
            print self.variables, 'selected variables'
            print self.signs, 'signs'
            print self.A.shape, 'A shape'
            print np.dot(self.A, Y).max(), 'should be nonpositive'

        self.P.append(Pnew)

    @property
    def constraints(self):
        if not hasattr(self, "_constraints"):
            if self.subset == []:
                self._constraints = constraints(self.A, np.zeros(self.A.shape[0]), 
                                                covariance=self.covariance)
            else:
                self._constraints = constraints(np.dot(self.A, self.subset_selector),
                                                np.zeros(self.A.shape[0]), 
                                                covariance=self.covariance)
        return self._constraints

    def model_pivots(self, which_step, alternative='greater',
                     saturated=True,
                     ndraw=5000,
                     burnin=2000,
                     which_var=[], 
                     compute_intervals=False,
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

        con = copy(self.constraints)
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
                        conditional_con = con.conditional(linear_part.T[keep],
                                                          observed[keep])
                    else:
                        conditional_con = con

                    if compute_intervals:
                        family = MLE_family(conditional_con,
                                            self.Y,
                                            LSfunc[i],
                                            burnin=burnin,
                                            ndraw=ndraw,
                                            how_often=50,
                                            white=False)

                        obs = (LSfunc[i] * self.Y).sum()
                        lower_lim, upper_lim = family.equal_tailed_interval(obs, 1 - coverage)

                        lower_lim_final = np.dot(LSfunc[i], np.dot(conditional_con.covariance, LSfunc[i])) * lower_lim
                        upper_lim_final = np.dot(LSfunc[i], np.dot(conditional_con.covariance, LSfunc[i])) * upper_lim

                        intervals.append((self.variables[i], (lower_lim_final, upper_lim_final)))
                        pval = family.cdf(0, obs)
                        pval = 2 * min(pval, 1 - pval)
                    else:
                        pval = gibbs_test(conditional_con,
                                          self.Y,
                                          LSfunc[i],
                                          alternative=alternative,
                                          sigma_known=sigma_known,
                                          burnin=burnin,
                                          ndraw=ndraw,
                                          how_often=-1)[0]
                    pivots.append((self.variables[i], 
                                   pval))
        return pivots

    def model_quadratic(self, which_step):
        LSfunc = np.linalg.pinv(self.X[:,self.variables[:which_step]])
        P_LS = np.linalg.svd(LSfunc, full_matrices=False)[2]
        return quadratic_test(self.Y, P_LS, self.constraints)

def canonicalA(RX, RY, idx, sign, scale=None):
    """
    The canonical set of inequalities for a step of forward stepwise.
    These encode that 
    `sign*np.dot(RX.T[idx],RY)=np.fabs(np.dot(RX,RY)).max()` which is
    $\|RX^TRY\|_{\infty}$.

    Parameters
    ----------

    RX : `np.array((n,p))`

    RY : `np.array(n)`

    idx : `int`
        Maximizing index of normalized `np.fabs(np.dot(RX.T,RY))` where normalization
        is left multiplication by a diagonal matrix
        represented  by `scale` and is generally such that each row of `RX.T` has $\ell_2$
        norm of 1. 

    sign : `[-1,1]`

    scale : `np.array(p)`
        A diagonal matrix to apply before computing the $\ell_{\infty}$ norm.

    """

    n, p = RX.shape

    if scale is None:
        scale = np.ones(p)

    A0 = np.vstack([np.diag(1./scale), np.diag(-1./scale)])
    v = np.zeros(p)
    v[idx] = sign/scale[idx]
    A = v[None,:] - A0

    U = np.dot(A0, np.dot(RX.T, RY))
    if DEBUG:
        if sign > 0:
            print np.fabs(U).max(), U[idx], 'should match'
        else:
            print np.fabs(U).max(), U[idx+p], 'should match'

    keep = np.ones(2*p, np.bool)
    if sign > 0:
        keep[idx] = 0
    else:
        keep[idx+p] = 0

    A = A[keep]

    V = np.dot(A, RX.T)
    return -V

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

    new_con = stack(FS.constraints, constraints(np.dot(new_linear_part, FS.subset_selector),
                                                new_offset))
    new_con.covariance[:] = sigma**2 * np.identity(n)
    if DEBUG:
        print FS.constraints.linear_part.shape, 'before'
    FS._constraints = new_con
    if DEBUG:
        print FS.constraints.linear_part.shape, 'should have added number of steps constraints'
    FS.active = FS.variables[:-1]
    return FS

def sequential(X, Y, sigma=None, nstep=10,
               saturated=False,
               ndraw=5000,
               burnin=2000,
               subset=[]):
    """
    Fit model using forward stepwise,
    stopping using a rule like AIC or BIC.
    
    The error variance must be supplied, in which
    case AIC is essentially Mallow's C_p.

    Parameters
    ----------

    X : np.float
        Design matrix

    Y : np.float
        Response vector

    sigma : float (optional)
        Error variance.

    nstep : int
        How many steps should we take?

    saturated : bool
        Should we compute saturated or selected model pivots?

    ndraw : int (optional)
        Defaults to 5000.

    burnin : int (optional)
        Defaults to 2000.

    subset : []
        Subset of cases to use for selection, defaults to [].

    Returns
    -------

    FS : `forward_stepwise`
        Instance of forward stepwise after `nstep` steps.

    pvalues : []
        P-values computed at each step.

    """

    n, p = X.shape
    if sigma is not None:
        FS = forward_stepwise(X, Y, covariance=sigma**2 * np.identity(n),
                              subset=subset)
    else:
        FS = forward_stepwise(X, Y)

    pvalues = []
    for i in range(nstep):
        FS.next()
        pvalues.extend(FS.model_pivots(i+1, which_var=[FS.variables[-1]],
                                       saturated=saturated,
                                       ndraw=ndraw,
                                       burnin=burnin))
    return FS, pvalues

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
