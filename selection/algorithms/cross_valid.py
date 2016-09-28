"""
Script to implement selective inference after cross-validation

"""

import numpy as np
from scipy.stats import norm as ndist

from regreg.api import identity_quadratic

from .lasso import lasso
from .sqrt_lasso import solve_sqrt_lasso, choose_lambda
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
               solve_args={'min_its':10, 'max_its':20},
               quadratic=None):
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
                                     quadratic=quadratic,
                                     solve_args=solve_args)[0]))
        else:
            results.append(
                (m, solve_sqrt_lasso(X, 
                                     Y, 
                                     m * L * np.ones(p), 
                                     quadratic=quadratic,
                                     initial=results[-1][1],
                                     solve_args=solve_args)[0]))

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
                       shift_size=0,
                       quadratic=None):
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

    quadratic : `regreg.identity_quadratic`
        A quadratic term added to objective function.

    """
    n, p = X.shape
    training = np.zeros(n, np.bool)
    training[np.random.choice(np.arange(n), size=int(test_frac*n), replace=False)] = 1
    test = ~training

    results = solve_grid(Y[training], X[training], L, mults=mults, quadratic=quadratic)
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

    kfold = sklearn.cross_validation.KFold(n=n, 
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
                      quadratic=None,
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
    SL = lasso.sqrt_lasso(X, Y, L * np.ones(p), quadratic=quadratic)
    SL.fit(solve_args=solve_args)
    return SL.active, SL.active_signs, SL

### end -- generalize from sqrt_lasso to smooth losses with \ell_1 penalty


## this class should be closer to examples in `selection.sampling.randomized` so
## we can reuse that code

class lasso_tuned(object):

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
                 randomization=ndist,
                 test_frac = 0.9,
                 mults = np.linspace(1.5,0.5,11),
                 sigma = None,
                 scale_inter = np.sqrt(0.2),
                 scale_select = np.sqrt(0.1),
                 scale_valid = np.sqrt(0.1),
                 shift_size=1):

        """

        Parameters
        ----------

        Y : np.float(n)
            Response vectors

        X : np.float((n,p))
            Design matrix.

        randomization : `scipy.stats.rv_continuous`
            A random variable with `pdf` and `rvs` methods.

        mults: [float]
            Sequence of floats over which to solve square-root LASSO.

        sigma : float
            Noise variance, if known. 

        scale_inter : float
            Proportion of variance (using
            `self.rough_sigma` as baseline) 
            added in randomization
            to Y_inter.

        scale_select : float
            Proportion of variance (using
            `self.rough_sigma` as baseline) 
            added in randomization
            to Y_select.

        scale_valid : float
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
         self.mults,
         self.randomization) = (
            Y, 
            X, 
            test_frac, 
            mults,
            randomization)

        self.L = choose_lambda(X)

        self.scale_inter = scale_inter
        self.scale_select = scale_select
        self.scale_valid = scale_valid

        # randomize our response

        self.randomize()

        # now find which CV values to accept

        self.accept_values = self.choose_lambda(self.Y, 
                                                shift_size=shift_size)
        self.selected_value = np.median(self.accept_values)
        self.choose_variables()

        self.null_sample = {}

        # estimate sigma if needed

        if sigma is not None:
            self.sigma_resid = sigma
        else:
            resid_current = (Y - np.dot(self.X[:,self.active_set],
                                        self.SQ.onestep_estimator))
            n = Y.shape[0]
            self.sigma_resid = np.linalg.norm(resid_current) / np.sqrt(n - self.active_set.shape[0])

        # find response independent of Y_inter, Y_valid, Y_select

        # XXX code below is specific to squared error loss -- need to rewrite for logistic
#         ratio = self.sigma_resid**2 / (self.scale_inter * self.rough_sigma)**2
#         self.Y_indep = Y - ratio * (self.Y_inter - Y)
#         self.betahat_indep = np.dot(np.linalg.pinv(self.X[:,self.active_set]), self.Y_indep)
#         cov_indep = np.linalg.pinv(np.dot(self.X[:,self.active_set].T, self.X[:,self.active_set])) * self.sigma_resid**2 * (1 + ratio)
#         T_indep = np.fabs(self.betahat_indep / np.sqrt(np.diag(cov_indep)))
#         self.pval_indep = 2 * (1 - ndist.cdf(T_indep))

    def randomize(self):
        """
        Carry out the randomization,
        finding the value of lambda
        as well as the selected variables and signs.

        Initiailizes the attributes: [Y_inter, Y_valid, Y_select].
        """

        n = self.Y.shape[0]

        # intermediate between 
        # CV and model selection 
        # and the actual data

        self.Q_inter = identity_quadratic(0, 0, self.randomization.rvs(size=self.X.shape[1]) * self.scale_inter, 0)
        self.Q_valid = self.Q_inter + identity_quadratic(0, 0, self.randomization.rvs(size=self.X.shape[1]) * self.scale_valid, 0) 
        self.Q_select = self.Q_inter + identity_quadratic(0, 0, self.randomization.rvs(size=self.X.shape[1]) * self.scale_select, 0)

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
                                  quadratic=self.Q_valid,
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
         self.SQ) = select_vars_signs(self.Y, 
                                      self.X,
                                      self.selected_value * self.L,
                                      quadratic=self.Q_select)

        self.inactive_set = self.SQ.inactive
        self._select_beta = self.SQ.lasso_solution
        self._select_loss = self.SQ.loglike
        self._select_subgrad = -(self._select_loss.smooth_objective(self._select_beta, 'grad') + 
                                 self.Q_select.objective(self._select_beta, 'grad'))

    def step_valid(self,
                   max_trials=10):
        """
        Try and move Y_valid
        by accept reject stopping after `max_trials`.
        """

        X, L, mults = self.X, self.L, self.mults
        n, p = X.shape

        count = 0
        Q_old = self.Q_valid

        while True:
            count += 1
            self.Q_valid = self.Q_inter + identity_quadratic(0, 0, self.randomization.rvs(size=self.X.shape[1]) * 
                                                             self.scale_valid, 0) 

            if len(self.mults) > 0:
                proposal_value = self.choose_lambda(self.Y,
                                                    shift_size=0)

                if proposal_value[0] in self.accept_values:
                    break
            else:
                break

            if count >= max_trials:
                self.Q_valid = Q_old
                break

    def step_select(self,
                    step_size=0.1):
        """
        Take `ndraw` Gibbs steps of Y_select
        """

        L_inter = self.Q_inter.linear_term
        L_select = self.Q_select.linear_term - L_inter
 
        # self.randomization defaults to Gaussian or beware!
        G_cur = np.linalg.norm(self._select_loss.smooth_objective(self._select_beta, 'grad') + 
                               L_inter + self._select_subgrad)**2 / self.scale_select**2

        while True:
            _beta = self._select_beta.copy()
            _beta[self.active_set] += (step_size * 
                                       self.randomization.rvs(size=self.active_set.shape) * 
                                       self.scale_select)

            _subgrad = self._select_subgrad.copy()
            _subgrad[self.inactive_set] += (step_size * 
                                            self.randomization.rvs(size=self.inactive_set.sum()) * 
                                            self.scale_select)


            if (np.all(np.sign(_beta) == np.sign(self._select_beta))
                and 
                np.all(np.fabs(_subgrad[self.inactive_set]) < self.SQ.feature_weights[self.inactive_set])):
                break

        G_proposal = np.linalg.norm(self._select_loss.smooth_objective(_beta, 'grad') + 
                                    L_inter + _subgrad)**2 / self.scale_select**2

        logMH_ratio = G_proposal - G_cur
        if np.random.sample() < np.exp(logMH_ratio): # MH step accepted
            self._select_beta[:] = _beta
            self._select_subgrad[:] = _subgrad

            self.Q_select.linear_term = -(self._select_loss.smooth_objective(_beta, 'grad') + 
                                          _subgrad)

    def step_inter(self,
                   do_gibbs=True):

        L_old = self.Q_inter.linear_term

        T_IS = self.Q_select.linear_term
        T_IV = self.Q_valid.linear_term

        quadratic_term = (1. / self.scale_inter**2 + 
                          1. / self.scale_valid**2 + 
                          1. / self.scale_select**2)

        linear_term = (T_IS / self.scale_select**2 + T_IV / self.scale_valid**2)

        sampling_sd = 1. / np.sqrt(quadratic_term)
        sampling_mean = linear_term / quadratic_term

        # self.randomization defaults to scipy.stats.norm -- otherwise beware!
        self.Q_inter.linear_term = (sampling_mean + self.randomization.rvs(size=T_IS.shape) * 
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
        Setup the current gaussian for sampling

        TODO: we should use the tilted distribution
        with the selectively unbiased estimate. Will help 
        with intervals.

        """
        p = self.X.shape[1]
        self._gaussian_mean = np.zeros(p)
        self._gaussian_cov = np.identity(p)
        self._invcov_noisy = 0.5 * np.identity(p)
        self._gaussian_conditional_sqrt = np.sqrt(0.5) * np.identity(p)
        self.which_var = which_var
        self.null_sample[which_var] = []
        self._gaussian_stat = np.zeros(p)
        self._gaussian_obs = self._gaussian_stat.copy()

    def step_sample(self):

        """
        Move Y_sample -- a Gaussian draw
        with mean depending on Y_inter.
        """

        p = self.X.shape[1]
        (mean, 
         cov, 
         invcov_noisy, 
         sampling_sqrt) = (self._gaussian_mean, 
                           self._gaussian_cov, 
                           self._invcov_noisy, 
                           self._gaussian_conditional_sqrt)

        noisy_statistic = self._gaussian_stat - self.Q_inter.linear_term
        sampling_mean = mean + cov.dot(invcov_noisy).dot(noisy_statistic - mean)
        self._gaussian_stat = sampling_mean + sampling_sqrt.dot(np.random.standard_normal(p))
        self.null_sample[self.which_var].append(self._gaussian_stat[self.which_var])

    def __iter__(self):
        if not hasattr(self, "which_var"):
            raise ValueError("choose a variable in active set on which to do inference")
        self.counter = 0
        return self

    def next(self):
        
        # move randomized responses Q_inter, Q_valid, Q_select
        self.step_randomized()

        # move Y_sample
        self.step_sample()
        
    __next__ = next # Python3 compatibility

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
        obs = self._gaussian_obs[self.which_var]
        pval = family.cdf(0, obs)
        pval = 2 * min(pval, 1 - pval)
    
        idx = list(self.active_set).index(which_var)
        return pval, self.pval_indep[idx]


class lasso_tuned_conditional(lasso_tuned):

    """
    Condition on the value of Y_valid -- accomplished by never
    sampling Y_valid.

    TODO: this can be made a fast sampler by automatically
    marginalizing over Y_inter.
    """

    CV_period = np.inf
    pass


