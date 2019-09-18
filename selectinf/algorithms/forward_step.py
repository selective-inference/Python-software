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

from ..constraints.affine import (constraints, 
                                  gibbs_test, 
                                  stack as stack_con,
                                  gaussian_hit_and_run)
from ..distributions.chain import parallel_test, serial_test
from ..distributions.chisq import quadratic_test
from ..distributions.discrete_family import discrete_family

DEBUG = False

class forward_step(object):

    """
    Forward stepwise model selection.

    """

    def __init__(self, X, Y, 
                 subset=None,
                 fixed_regressors=None,
                 intercept=True,
                 covariance=None):

        """
        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        Y : ndarray
            Shape (n,) -- the response.

        subset : ndarray (optional)
            Shape (n,) -- boolean indicator of which cases to use.
            Defaults to np.ones(n, np.bool)

        fixed_regressors: ndarray (optional)
            Shape (n, *) -- fixed regressors to regress out before
            computing score.

        intercept : bool
            Remove intercept -- this effectively includes np.ones(n) to fixed_regressors.

        covariance : ndarray (optional)
            Covariance matrix of errors. Defaults to np.identity(n).

        Returns
        -------

        FS : `selection.algorithms.forward_step.forward_step`
        
        Notes
        -----

        """

        self.subset = subset
        self.X, self.Y = X, Y

        n, p = self.X.shape
        if fixed_regressors is not None:
            fixed_regressors = np.asarray(fixed_regressors).reshape((n,-1))

        if intercept:
            if fixed_regressors is not None:
                fixed_regressors = np.hstack([fixed_regressors, np.ones((n, 1))])
            else:
                fixed_regressors = np.ones((n, 1))

        if fixed_regressors is not None:
            self.fixed_regressors = np.hstack(fixed_regressors)
            if self.fixed_regressors.ndim == 1:
                self.fixed_regressors = self.fixed_regressors.reshape((-1,1))

            # regress out the fixed regressors
            # TODO should be fixed for subset
            # should we adjust within the subset or not?

            self.fixed_pinv = np.linalg.pinv(self.fixed_regressors)
            self.Y = self.Y - np.dot(self.fixed_regressors, 
                                     np.dot(self.fixed_pinv, self.Y))
            self.X = self.X - np.dot(self.fixed_regressors, 
                                     np.dot(self.fixed_pinv, self.X))
        else:
            self.fixed_regressors = None

        if self.subset is not None:

            self.working_X = self.X.copy()[subset]
            self.subset_X = self.X.copy()[subset]
            self.subset_Y = self.Y.copy()[subset]
            self.subset_selector = np.identity(self.X.shape[0])[subset]
            self.subset_fixed = self.fixed_regressors[subset]
        else:
            self.working_X = self.X.copy()
            self.subset_Y = self.Y.copy()
            self.subset_X = self.X.copy()
            self.subset_fixed = self.fixed_regressors

        # scale columns of X to have length 1
        self.working_X /= np.sqrt((self.working_X**2).sum(0))[None, :]

        self.variables = [] # the sequence of selected variables
        self.Z = []         # the achieved Z scores
        self.Zfunc = []     # the linear functionals of Y that achieve the Z scores
        self.signs = []     # the signs of the achieved Z scores

        self.covariance = covariance               # the covariance of errors
        self._resid_vector = self.subset_Y.copy()  # the current residual -- already adjusted for fixed regressors

        # setup for iteration

        self.identity_constraints = []    # this will store linear functionals that identify the variables
        self.inactive = np.ones(p, np.bool)   # current inactive set
        self.maxZ_offset = np.array([np.ones(p) * np.inf, np.ones(p) * np.inf]) # stored for computing
                                                                                # the limits of maxZ selected test
        self.maxZ_constraints = []

    def step(self, 
             compute_maxZ_pval=False,
             use_identity=False,
             ndraw=8000,
             burnin=2000,
             sigma_known=True,
             accept_reject_params=(100, 15, 2000)):
        """
        Parameters
        ----------

        compute_maxZ_pval : bool
            Compute a p-value for this step? Requires MCMC sampling.

        use_identity : bool
            If computing a p-value condition on the identity of the variable?

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        sigma_known : bool
            Is $\sigma$ assumed known?

        accept_reject_params : tuple
            If not () should be a tuple (num_trial, min_accept, num_draw).
            In this case, we first try num_trial accept-reject samples,
            if at least min_accept of them succeed, we just draw num_draw
            accept_reject samples.

        """
        
        working_X, Y = self.working_X, self.subset_Y
        resid_vector = self._resid_vector
        n, p = working_X.shape

        # up to now inactive
        inactive = self.inactive

        # compute Z scores

        scale = self.scale = np.sqrt(np.sum(working_X**2, 0))
        scale[~inactive] = np.inf # should never be used in any case
        Zfunc = working_X.T # [inactive] 
        Zstat = np.dot(Zfunc, Y) / scale # [inactive]

        winning_var = np.argmax(np.fabs(Zstat))
        winning_sign = np.sign(Zstat[winning_var])
        winning_func = Zfunc[winning_var] / scale[winning_var] * winning_sign

        realized_maxZ = Zstat[winning_var] * winning_sign 
        self.Z.append(realized_maxZ)

        if self.subset is not None:
            self.Zfunc.append(winning_func.dot(self.subset_selector))
        else:
            self.Zfunc.append(winning_func)

        # keep track of identity for testing
        # variables other than the last one added

        # this adds a constraint to self.identity_constraints

        # losing_vars are variables that are inactive (i.e. not in self.variables)
        # and did not win in this step

        losing_vars = inactive.copy()
        losing_vars[winning_var] = False

        identity_linpart = np.vstack([ 
                working_X[:,losing_vars].T / scale[losing_vars,None] -
                winning_func,
                -working_X[:,losing_vars].T / scale[losing_vars,None] -
                winning_func,
                - winning_func.reshape((1,-1))])

        if self.subset is not None:
            identity_linpart = np.dot(identity_linpart, 
                                      self.subset_selector)

        identity_con = constraints(identity_linpart,
                                   np.zeros(identity_linpart.shape[0]))

        if not identity_con(self.Y):
            raise ValueError('identity fail!')

        self.identity_constraints.append(identity_linpart)

        # form the maxZ constraint

        XI = self.subset_X[:,self.inactive]
        linear_part = np.vstack([XI.T, -XI.T])
        if self.subset is not None:
            linear_part = np.dot(linear_part, 
                                 self.subset_selector)

        inactive_offset = self.maxZ_offset[:, self.inactive]

        maxZ_con = constraints(linear_part, np.hstack(inactive_offset),
                               covariance=self.covariance)

        if use_identity:
            maxZ_con = stack_con(maxZ_con, identity_con)
            maxZ_con.covariance = self.covariance

        if len(self.variables) > 0 or (self.fixed_regressors != []):
            XA = self.subset_X[:, self.variables]
            XA = np.hstack([self.subset_fixed, XA])
            # the RHS, i.e. offset is fixed by this conditioning
            if self.subset is not None:
                conditional_con = maxZ_con.conditional(XA.T.dot(self.subset_selector),
                                                       np.dot(XA.T, Y))
            else:
                conditional_con = maxZ_con.conditional(XA.T,
                                                       np.dot(XA.T, Y))
        else:
            conditional_con = maxZ_con

        self.maxZ_constraints.append(conditional_con)
        if compute_maxZ_pval:
            maxZ_pval = self._maxZ_test(ndraw, burnin,
                                        sigma_known=sigma_known,
                                        accept_reject_params=accept_reject_params)

        # now update for next step

        # update the offsets for maxZ

        # when we condition on the sufficient statistics up to
        # and including winning_var, the Z_scores are fixed
        
        # then, the losing variables at this stage can be expressed as
        # abs(working_X.T.dot(Y)[:,inactive] / scale[inactive]) < realized_maxZ
        # where inactive is the updated inactive 

        # the event we have witnessed this step is 
        # $$\|X^T_L(I-P)Y / diag(X^T_L(I-P)X_L)\|_{\infty} \leq X^T_W(I-P)Y / \sqrt(X^T_W(I-P)X_W)$$
        # where P is the current "model"

        # let V=PY and S_L the losing scales, we rewrite this as
        # $$\|(X^T_LY - V) / S_L\|_{\infty} \leq Z_max $$
        # and again
        # $$X^T_LY / S_L - V / S_L \leq Z_max, -(X^T_LY / S_L - V / S_L) \leq Z_max $$
        # or,
        # $$X^T_LY \leq Z_max * S_L + V, -X^T_LY \leq Z_max * S_L - V $$

        # where, at the next step Z_max and V are measurable with respect to
        # the appropriate sigma algebra

        realized_Z_adjustment = realized_maxZ * scale                      # Z_max * S_L
        fit_adjustment = np.dot(self.subset_X.T, Y - resid_vector)         # V * S_L
        self.maxZ_offset[0] = np.minimum(self.maxZ_offset[0], realized_Z_adjustment + fit_adjustment)   # (Z_max + V) * S_L
        self.maxZ_offset[1] = np.minimum(self.maxZ_offset[1], realized_Z_adjustment - fit_adjustment)  # (Z_max - V) * S_L

        # update our list of variables and signs

        self.inactive[winning_var] = False # inactive is now losing_vars
        self.variables.append(winning_var); self.signs.append(winning_sign)

        # update residual, and adjust X

        resid_vector -= realized_maxZ * winning_func
        working_X -= (np.multiply.outer(winning_func, winning_func.dot(working_X)) /
                       (winning_func**2).sum())

        if compute_maxZ_pval:
            return maxZ_pval

    def constraints(self, step=np.inf, identify_last_variable=True):
        default_step = len(self.variables)
        if default_step > 0 and not identify_last_variable:
            default_step -= 1
        step = min(step, default_step)
        A = np.vstack(self.identity_constraints[:step])

        con = constraints(A, 
                          np.zeros(A.shape[0]), 
                          covariance=self.covariance)
        return con

    def _maxZ_test(self, 
                   ndraw, 
                   burnin,
                   sigma_known=True,
                   accept_reject_params=(100, 15, 2000)
                   ):

        XI, Y = self.subset_X[:, self.inactive], self.subset_Y
        sequential_con = self.maxZ_constraints[-1]
        if not sequential_con(Y):
            raise ValueError('Constraints on Y not satisfied')

        # use partial
        def maxT(Z, L=self.working_X[:,self.inactive], S=self.scale[self.inactive]):
            Tstat = np.fabs(np.dot(Z, L) / S[None,:]).max(1)
            return Tstat

        pval, _, _, dfam = gibbs_test(sequential_con,
                                      Y,
                                      self.Zfunc[-1],
                                      sigma_known=sigma_known,
                                      white=False,
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      how_often=-1,
                                      UMPU=False,
                                      use_random_directions=False,
                                      tilt=None,
                                      alternative='greater',
                                      test_statistic=maxT,
                                      accept_reject_params=accept_reject_params
                                      )
        return pval

    def model_pivots(self, which_step, 
                     alternative='onesided',
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

        alternative : ['onesided', 'twosided']
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

        if alternative not in ['onesided', 'twosided']:
            raise ValueError('alternative should be either "onesided" or "twosided"')

        if which_step == 0:
            return []

        if self.covariance is None and saturated:
            raise ValueError('need a covariance matrix to compute pivots for saturated model')

        con = copy(self.constraints(which_step))

        if self.covariance is not None:
            con.covariance = self.covariance 

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
                    if alternative == 'onesided':
                        _alt = {1:'greater',
                                -1:'less'}[self.signs[i]]
                    else:
                        _alt = 'twosided'
                    pivots.append((self.variables[i],
                                   con.pivot(LSfunc[i], self.Y,
                                             alternative=_alt)))
                  
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

                        if alternative == 'onesided':
                            _alt = {1:'greater',
                                    -1:'less'}[self.signs[i]]
                        else:
                            _alt = 'twosided'

                        _ , _, _, family = gibbs_test(conditional_law,
                                                      self.Y,
                                                      eta,
                                                      sigma_known=True,
                                                      white=False,
                                                      ndraw=ndraw,
                                                      burnin=burnin,
                                                      how_often=10,
                                                      use_random_directions=False,                                                     
                                                      UMPU=False,
                                                      alternative=_alt)

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
        return quadratic_test(self.Y, P_LS, self.constraints(step=which_step))

def info_crit_stop(Y, X, sigma, cost=2,
                   subset=None):
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

    subset : ndarray (optional)
        Shape (n,) -- boolean indicator of which cases to use.
        Defaults to np.ones(n, np.bool)

    Returns
    -------

    FS : `forward_step`
        Instance of forward stepwise stopped at the
        corresponding step. Constraints of FS
        will reflect the minimum Z score requirement.

    """
    n, p = X.shape
    FS = forward_step(X, Y, covariance=sigma**2 * np.identity(n), subset=subset)

    while True:
        FS.step()
        if FS.Z[-1] < sigma * np.sqrt(cost):
            break

    new_linear_part = -np.array(FS.Zfunc)
    new_linear_part[-1] *= -1
    new_offset = -sigma * np.sqrt(cost) * np.ones(new_linear_part.shape[0])
    new_offset[-1] *= -1

    new_con = stack_con(FS.constraints(), constraints(new_linear_part,
                                                      new_offset))
    new_con.covariance[:] = sigma**2 * np.identity(n)
    FS._constraints = new_con
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

def mcmc_test(fs_obj, step, variable=None,
              nstep=100,
              ndraw=20,
              method='parallel', 
              burnin=1000,):

    if method not in ['parallel', 'serial']:
        raise ValueError("method must be in ['parallel', 'serial']")

    X, Y = fs_obj.subset_X, fs_obj.subset_Y

    variables = fs_obj.variables[:step]

    if variable is None:
        variable = variables[-1]

    if variable not in variables:
        raise ValueError('variable not included at given step')

    A = np.vstack(fs_obj.identity_constraints[:step])
    con = constraints(A, 
                      np.zeros(A.shape[0]), 
                      covariance=fs_obj.covariance)

    XA = X[:,variables]
    con_final = con.conditional(XA.T, XA.T.dot(Y))

    if burnin > 0:
        chain_final = gaussian_hit_and_run(con_final, Y, nstep=burnin)
        chain_final.step()
        new_Y = chain_final.state
    else:
        new_Y = Y

    keep = np.ones(XA.shape[1], np.bool)
    keep[list(variables).index(variable)] = 0
    nuisance_variables = [v for i, v in enumerate(variables) if keep[i]]

    if nuisance_variables:
        XA_0 = X[:,nuisance_variables]
        beta_dir = np.linalg.solve(XA_0.T.dot(XA_0), XA_0.T.dot(X[:,variable]))
        adjusted_direction = X[:,variable] - XA_0.dot(beta_dir)
        con_test = con.conditional(XA_0.T, XA_0.T.dot(Y))
    else:
        con_test = con
        adjusted_direction = X[:,variable]

    chain_test = gaussian_hit_and_run(con_test, new_Y, nstep=nstep)
    test_stat = lambda y: -np.fabs(adjusted_direction.dot(y))

    if method == 'parallel':
        rank = parallel_test(chain_test,
                             new_Y,
                             test_stat)
    else:
        rank = serial_test(chain_test,
                           new_Y,
                           test_stat)

    return rank
