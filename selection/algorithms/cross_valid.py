"""
Script to implement selective inference after cross-validation
 

"""

import numpy as np
from scipy.stats import norm as ndist

from .sqrt_lasso import sqrt_lasso, solve_sqrt_lasso, choose_lambda
from ..constraints.affine import (constraints, 
                                  sample_from_constraints)
from ..distributions.discrete_family import discrete_family

# These next few functions should be generalized to not
# be just sqrt_lasso

### begin -- generalize from sqrt_lasso to smooth losses with \ell_1 penalty

def solve_grid(Y, 
               X, 
               L, 
               mults, 
               post_estimator=False,
               solve_args={'min_its':10, 'max_its':20}):
    """
    Solve the square-root LASSO over a grid of values.

    .. math::

        \text{minimize}_{\beta} \|y-X\beta\|_2 + m * L \|\beta\|_1

    for $m$ in `mults`.

    Parameters
    ----------

    Y : np.float(n)
        Response vectors

    X : np.float((n,p))
        Design matrix.

    L : float
        Value of $\lambda$ in square-root LASSO optimization
        problem.

    mults: [float]
        Sequence of floats over which to solve square-root LASSO.

    post_estimator: bool
        Should we return the square-root LASSO estimate or the
        OLS of the selected model (the post square-root LASSO estimator).

    solve_args : {}
        Keyword arguments passed to `solve_sqrt_lasso`.

    Returns
    -------

    results : [(m, beta_m)]
        Coefficient estimates for each `m` in `mults`.

    """
    n, p = X.shape
    results = []
    for i, m in enumerate(mults):
        if i == 0:
            results.append(
                (m, solve_sqrt_lasso(X, 
                                     Y, 
                                     m * L * np.ones(p), 
                                     **solve_args)))
        else:
            results.append(
                (m, solve_sqrt_lasso(X, 
                                     Y, 
                                     m * L * np.ones(p), 
                                     initial=results[-1][1],
                                     **solve_args)))

        if post_estimator:
            active = np.nonzero(results[-1][1])[0]
            coef = np.zeros(p)
            if active.shape[0] > 0:
                X_E = X[:,active]
                coef[active] = np.dot(np.linalg.pinv(X_E), Y)
            results[-1] = (m, coef)

    return results

def split_and_validate(Y, 
                       X, 
                       L, 
                       mults, 
                       test_frac,
                       shift_size=0):
    """
    Choose which lambda minimizes prediction
    over a random split.

    Parameters
    ----------

    Y : np.float(n)
        Response vectors

    X : np.float((n,p))
        Design matrix.

    L : float
        Value of $\lambda$ in square-root LASSO optimization
        problem.

    mults: [float]
        Sequence of floats over which to solve square-root LASSO.

    test_frac: float
        What percentage should be used as test?

    shift_size : int
        Return minimizer plus a uniform 
        positive or negative shift in the index 
        of `mults` of a given size.
        Affects the size of the window of 
        minimizers to be accepted by later sampling scheme.

    """
    n, p = X.shape
    training = np.zeros(n, np.bool)
    training[np.random.choice(np.arange(n), size=int(test_frac*n), replace=False)] = 1
    test = ~training

    results = solve_grid(Y[training], X[training], L, mults=mults)
    error = []
    for m, coef in results:
        error.append((np.linalg.norm(Y[test] - np.dot(X[test], coef))**2, m))
    m_min = min(error)[1]
    idx_min = list(mults).index(m_min)
    
    # this shift randomizes the returned value of \lambda
    # have not really used it much.
    
    if shift_size > 0:
        random_shift = np.random.random_integers(low=-shift_size,
                                          high=shift_size)
        idx_min += random_shift
        idx_min = max(idx_min, 0)
    return [mults[idx_min + j] for j in range(-shift_size, shift_size+1, 1)
            if idx_min + j >= 0 and idx_min + j < len(mults)]

def kfold_CV(Y, 
             X, 
             L, 
             mults, 
             K=10,
             random_shift=0,
             shuffle=True, random_state=False):
    """
    Choose which lambda minimizes prediction
    using K-fold cross-validation.


    Parameters
    ----------

    Y : np.float(n)
        Response vectors

    X : np.float((n,p))
        Design matrix.

    L : float
        Value of $\lambda$ in square-root LASSO optimization
        problem.

    mults: [float]
        Sequence of floats over which to solve square-root LASSO.

    K : int
        Number of folds (defaults to 10).

    shift_size : int
        Return minimizer plus a uniform 
        positive or negative shift in the index 
        of `mults` of a given size.
        Affects the size of the window of 
        minimizers to be accepted by later sampling scheme.

    shuffle : bool
        Argument to `sklearn.cross_validation.KFold`

    random_state : None, int or RandomState
        Argument to `sklearn.cross_validation.KFold`
    
    Returns
    -------

    window : [float]
        Values of multiplier that will be accepted
        in sampling routine.

    """

    n, p = X.shape

    kfold = cross_validation.KFold(n=n, 
                                   n_folds=K, 
                                   shuffle=shuffle,
                                   random_state=random_state)
    error = {}

    for train_index, test_index in kfold:
        results = solve_grid(Y[train_index], X[train_index], L, mults=mults)
        for m, coef in results:
            error.setdefault(m, []).append(
                nplinalg.norm(Y[test_index] - np.dot(X[test_index], coef))**2)
    
    for m in mults:
        error[m] = (np.mean(error[m]), np.std(error[m]))
    m_min = min([(error[k], k) for k in error])[1]
    idx_min = list(mults).index(m_min)
    if shift_size > 0:
        random_shift = np.random.random_integers(low=-shift_size,
                                          high=shift_size)
        idx_min += random_shift
        idx_min = max(idx_min, 0)
    return [mults[idx_min + j] for j in range(-shift_size, shift_size+1, 1)
            if idx_min + j >= 0 and idx_min + j < len(mults)]

def select_vars_signs(Y, 
                      X, 
                      L, 
                      solve_args={'min_its':150}):

    """
    Return active set and signs for solution
    of square-root LASSO.

    Parameters
    ----------

    Y : np.float(n)
        Response vectors

    X : np.float((n,p))
        Design matrix.

    L : float
        Value of $\lambda$ in square-root LASSO optimization
        problem.

    solve_args : {}
        Keyword arguments passed to `solve_sqrt_lasso`.

    Returns
    -------

    active : [int] 
        Active set.

    signs : [-1,1]
        Signs of variables in active set.

    sqlasso : `selection.algorithms.sqrt_lasso.sqrt_lasso`
        Instance whose signs and active sets we return.

    """
    n, p = X.shape
    SL = sqrt_lasso(Y, X, L * np.ones(p))
    SL.fit(**solve_args)
    return SL.active, SL.z_E, SL

### end -- generalize from sqrt_lasso to smooth losses with \ell_1 penalty


## this class should be closer to examples in `selection.sampling.randomized` so
## we can reuse that code

class sqrt_lasso_tuned(object):

    """
    
    Selective inference after choosing lambda
    in sqrt LASSO.
    
    Uses selected model on randomized data
    after having chosen \lambda.

    When \sigma^2_E is unknown
    we estimate \sigma^2_E.

    """

    CV_period = 50 # how often to try to update Y_CV

    def __init__(self, 
                 Y, 
                 X,
                 mults = np.linspace(1.5,0.5,11),
                 target_R2 = 0.5,
                 test_frac = 0.9,
                 sigma = None,
                 sd_inter = np.sqrt(0.2),
                 sd_select = np.sqrt(0.1),
                 sd_valid = np.sqrt(0.1),
                 shift_size=1
                 ):

        """

        Parameters
        ----------

        Y : np.float(n)
            Response vectors

        X : np.float((n,p))
            Design matrix.

        mults: [float]
            Sequence of floats over which to solve square-root LASSO.

        target_R2 : float
            Rough guess at population $R^2$. Used 
            to find a rough estimate of noise variance
            if not known.

        sigma : float
            Noise variance, if known. 

        sd_inter : float
            Proportion of variance (using
            `self.rough_sigma` as baseline) 
            added in randomization
            to Y_inter.

        sd_select : float
            Proportion of variance (using
            `self.rough_sigma` as baseline) 
            added in randomization
            to Y_select.

        sd_valid : float
            Proportion of variance (using
            `self.rough_sigma` as baseline) 
            added in randomization
            to Y_valid.

        shift_size : int
            Return minimizer plus a uniform 
            positive or negative shift in the index 
            of `mults` of a given size.
            Affects the size of the window of 
            minimizers to be accepted by later sampling scheme.

        """
        n, p = X.shape

        (self.Y, 
         self.X, 
         self.test_frac, 
         self.mults) = (
            Y, 
            X, 
            test_frac, 
            mults)

        self.L = choose_lambda(X)

        self.sd_inter = sd_inter
        self.sd_select = sd_select
        self.sd_valid = sd_valid

        # get a rough estimate of sigma
        # based on a rough population
        # guess of R^2

        if sigma is None:
            self.rough_sigma = np.sqrt((1 - target_R2) * np.linalg.norm(Y)**2 / n)
        else:
            self.rough_sigma = sigma

        # randomize our response

        self.randomize()

        # now find which CV values to accept

        self.accept_values = self.choose_lambda(self.Y_valid, 
                                                shift_size=shift_size)

        # TODO: there is a boundary issue above
        # if we actually randomize lambda
        # no issue when shift_size=0

        self.selected_value = np.median(self.accept_values)
        self.choose_variables()

        self.null_sample = {}

        # estimate sigma if needed

        if sigma is not None:
            self.sigma_resid = sigma
        else:
            resid_current = (Y - np.dot(self.X[:,self.active_set],
                                        np.dot(self.SQ._XEinv, Y)))
            n = Y.shape[0]
            self.sigma_resid = np.linalg.norm(resid_current) / np.sqrt(n - self.active_set.shape[0])

        # find response independent of Y_inter, Y_valid, Y_select

        ratio = self.sigma_resid**2 / (self.sd_inter * self.rough_sigma)**2
        self.Y_indep = Y - ratio * (self.Y_inter - Y)
        self.betahat_indep = np.dot(np.linalg.pinv(self.X[:,self.active_set]), self.Y_indep)
        cov_indep = np.linalg.pinv(np.dot(self.X[:,self.active_set].T, self.X[:,self.active_set])) * self.sigma_resid**2 * (1 + ratio)
        T_indep = np.fabs(self.betahat_indep / np.sqrt(np.diag(cov_indep)))
        self.pval_indep = 2 * (1 - ndist.cdf(T_indep))

    def randomize(self):
        """
        Carry out the randomization,
        finding the value of lambda
        as well as the selected variables and signs.

        Initiailizes the attributes: [Y_inter, Y_valid, Y_select].
        """

        n = self.Y.shape[0]

        self.Y_sample = self.Y.copy()

        # intermediate between 
        # CV and model selection 
        # and the actual data

        self.Y_inter = self.Y_sample + (self.rough_sigma * 
                                        np.random.standard_normal(n) *
                                        self.sd_inter)

        # used for choosing CV
        self.Y_valid = self.Y_inter + (self.rough_sigma * 
                                    np.random.standard_normal(n) *
                                    self.sd_valid) 

        # used for choosing variables and signs

        self.Y_select = self.Y_inter + (self.rough_sigma * 
                                        np.random.standard_normal(n) *
                                        self.sd_select) 

    def choose_lambda(self, Y, shift_size=0):
        """
        Select a value of lambda using `self.Y_valid`

        Stores result in attribute `accept_values`.

        Any resampling of Y_valid that results in a value within these
        values has a chance to be accepted.

        Parameters
        ----------

        Y : np.float(n)
            Response vector.

        shift_size : int
            Return minimizer plus a uniform 
            positive or negative shift in the index 
            of `mults` of a given size.
            Affects the size of the window of 
            minimizers to be accepted by later sampling scheme.

        """
        return split_and_validate(Y,
                                  self.X,
                                  self.L, 
                                  self.mults, 
                                  self.test_frac,
                                  shift_size=shift_size)
        
    def choose_variables(self):
        """
        Select variables and signs `self.Y_select`

        Stores results in attributes `(active_set, active_signs)`.

        Also initializes some attributes used in sampling Y_select.
        """
        # now, select a model

        (self.active_set, 
         self.active_signs,
         self.SQ) = select_vars_signs(self.Y_select, 
                                      self.X,
                                      self.selected_value * self.L)

        offset = self.SQ.active_constraints.offset
        linear_part = - np.identity(offset.shape[0])

        self.X_E = self.X[:,self.active_set]
        self.OLS_matrix = self.SQ._XEinv
        self.coef_select = (np.dot(self.OLS_matrix, self.Y_select) *
                               self.SQ.z_E)

        self.constraints = constraints(linear_part, offset)
        self.constraints.mean[:] = (np.dot(self.OLS_matrix, self.Y_inter) *
                                    self.SQ.z_E)
        self.constraints.covariance[:] = (np.dot(self.OLS_matrix,
                                          self.OLS_matrix.T) *
                                    self.rough_sigma**2 * self.sd_select**2)

    def step_valid(self,
                max_trials=10):
        """
        Try and move Y_valid
        by accept reject stopping after `max_trials`.
        """

        X, L, mults = self.X, self.L, self.mults
        n, p = X.shape

        count = 0
        while True:
            count += 1
            Y_proposal = self.Y_inter + (np.random.standard_normal(n) 
                                      * self.sd_valid * self.rough_sigma)

            if len(self.mults) > 0:
                proposal_value = self.choose_lambda(Y_proposal,
                                                    shift_size=0)

                if proposal_value[0] in self.accept_values:
                    self.Y_valid[:] = Y_proposal
                    break
            else:
                self.Y_valid[:] = Y_proposal
                break

            if count >= max_trials:
                break

    def step_select(self,
                    ndraw=500,
                    fix_residual=True):
        """
        Take `ndraw` Gibbs steps of Y_select
        """

        self.constraints.mean[:] = (np.dot(self.OLS_matrix, self.Y_inter) *
                                    self.SQ.z_E)
        Y_current = self.Y_select.copy()
        sample = sample_from_constraints(self.constraints,
                                         self.coef_select,
                                         self.coef_select,
                                         ndraw=ndraw,
                                         burnin=0)
        self.coef_select[:] = sample[-1]
        Y_hat = np.dot(self.X_E, self.active_signs * self.coef_select)
        self.Y_select += Y_hat - Y_current

    def step_inter(self,
                   do_gibbs=True):
        quadratic_term = (1. / self.sd_inter**2 + 
                          1. / self.sd_valid**2 + 
                          1. / self.sd_select**2)

        sampling_sd = self.rough_sigma * 1. / np.sqrt(quadratic_term)
        sampling_mean = ((self.Y_sample / self.sd_inter**2 + 
                          self.Y_valid / self.sd_valid**2 + 
                          self.Y_select / self.sd_select**2) / 
                         quadratic_term)
        n = self.Y_sample.shape[0]

        self.Y_inter[:] = (sampling_mean + np.random.standard_normal(n) * 
                           sampling_sd)
        
    def step_randomized(self):
        """
        Take a move on the all 
        randomized variables.
        """

        self.counter += 1

        if self.counter % self.CV_period == 0:
            self.step_valid()
        
        self.step_select()
        self.step_inter()

    def setup_inference(self, which_var): 
        """
        Setup sampling to sample from
        null distribution for a given variable.

        TODO: we should use the tilted distribution
        with the selectively unbiased estimate. Will help 
        with intervals.

        """
        self.which_var = which_var
        which_idx = list(self.active_set).index(which_var)
        keep = np.ones(self.active_set.shape[0], np.bool)
        keep[which_idx] = False
        self._X_Ej = self.X_E[:,keep]
        self._X_j = self.X[:,which_var]
        self._X_Eji = np.linalg.pinv(self._X_Ej)

        self.null_sample.setdefault(which_var, [])

        # maybe we should reinitialize
        # self.Y_sample[:] = self.Y

        self._mu_j = np.dot(self._X_Ej, 
                            np.dot(self._X_Eji, self.Y))
    def step_sample(self):

        """
        Move Y_sample -- a Gaussian draw
        with mean depending on Y_inter.
        """

        n, p = self.X.shape
        self.null_sample[self.which_var].append((self._X_j * self.Y_sample).sum())
        sigma_resid = self.sigma_resid

        quadratic_term = 1. / sigma_resid**2 + 1. / (self.rough_sigma * self.sd_inter)**2
        sampling_sd = 1. / np.sqrt(quadratic_term)

        sampling_mean = (self.Y_inter * 1. / 
                         (self.rough_sigma * self.sd_inter)**2) / quadratic_term
        
        uncond_draw = sampling_mean + (np.random.standard_normal(n) * 
                                       sampling_sd)
        proj_draw = uncond_draw - np.dot(self._X_Ej,
                                         np.dot(self._X_Eji, uncond_draw))
        self.Y_sample[:] = self._mu_j + proj_draw

    def __iter__(self):
        if not hasattr(self, "which_var"):
            raise ValueError("choose a variable in active set on which to do inference")
        self.counter = 0
        return self

    def next(self):
        
        # move randomized responses Y_inter, Y_valid, Y_select
        self.step_randomized()

        # move Y_sample
        self.step_sample()
        
    def pvalue(self, which_var,
               ndraw=2000,
               burnin=500):
        """
        Produce two p-values for one of the
        active variables, which_var, assumed to be in self.active_set

        First one uses sampling, the second based on
        a particular conditional distribution.
        """

        self.setup_inference(which_var); iter(self)
        for _ in xrange(ndraw + burnin):
            self.next()

        family = discrete_family(self.null_sample[which_var][burnin:],
                                 np.ones(ndraw))
        obs = (self._X_j * self.Y).sum()
        pval = family.cdf(0, obs)
        pval = 2 * min(pval, 1 - pval)
    
        idx = list(self.active_set).index(which_var)
        return pval, self.pval_indep[idx]


class sqrt_lasso_tuned_conditional(sqrt_lasso_tuned):

    """
    Condition on the value of Y_valid -- accomplished by never
    sampling Y_valid.

    TODO: this can be made a fast sampler by automatically
    marginalizing over Y_inter.
    """

    CV_period = np.inf
    pass


