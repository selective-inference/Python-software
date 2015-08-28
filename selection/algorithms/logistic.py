"""
This module contains a class `logistic`_ that implements
post selection inference for logistic LASSO.

"""

import warnings
from copy import copy

import numpy as np

from regreg.api import logistic_loss, weighted_l1norm, simple_problem, identity_quadratic

from selection.constraints.affine import (constraints, selection_interval,
                                 interval_constraints,
                                 sample_from_constraints,
                                 gibbs_test,
                                 stack)
from selection.distributions.discrete_family import discrete_family
from selection.algorithms.lasso import lasso as OLS_lasso, _constraint_from_data

class lasso(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta;X,Y) + 
             \lambda \|\beta\|_1

    where $\lambda$ is `lam` and $\ell$ is logistic loss.


    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    randomization_sd = 0.3

    def __init__(self, y, X, weights,
                 trials=None,
                 randomization_covariance=None):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        y : np.float(n)
            The response vector assumed to be a proportion, with
            each entry based on Binomial with a given number
            of trials.

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        weights : np.float(p) or float
            Coefficients in weighted L-1 penalty in
            optimization problem. If a float,
            weights are proportional to 1.

        trials : np.int(n) (optional)
            Number of trials for each of the proportions.
            If not specified, defaults to np.ones(n)

        """
        self.y, self.X = y, X
        n, p = X.shape

        if np.array(weights).shape == ():
            weights = weights * np.ones(p)
        self.weights = weights

        if trials is None:
            trials = np.ones_like(y)
        self.trials = trials
        self._successes = self.trials * self.y
        n, p = X.shape

        if randomization_covariance is None:
            randomization_covariance = np.identity(p) * self.randomization_sd**2
        self.randomization_covariance = randomization_covariance
        self.randomization_chol = np.linalg.cholesky(self.randomization_covariance)
        self.random_linear_term = np.dot(self.randomization_chol,
                                         np.random.standard_normal(p))

    def fit(self, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fit the lasso using `regreg`.
        This sets the attribute `soln` and
        forms the constraints necessary for post-selection inference
        by calling `form_constraints()`.

        Parameters
        ----------

        solve_args : {}
             Passed to `regreg.simple_problem.solve``_

        Returns
        -------

        soln : np.float
             Solution to lasso with `sklearn_alpha=self.lagrange`.
             
        """

        n, p = self.X.shape
        self.loss = logistic_loss(self.X, self.y * self.trials, trials=self.trials,
                                  coef=n/2.)
        self.penalty = weighted_l1norm(self.weights, lagrange=1.)
        self.penalty.quadratic = identity_quadratic(0, 0, self.random_linear_term, 0)
        problem = simple_problem(self.loss, self.penalty)
        soln = problem.solve(**solve_args)

        self._soln = soln
        if not np.all(soln == 0):
            self.active = np.nonzero(soln)[0]
            self.inactive = np.array(sorted(set(xrange(p)).difference(self.active)))
            X_E = self.X[:,self.active]
            loss_E = logistic_loss(X_E, self.y * self.trials, trials=self.trials)
            self._beta_unpenalized = loss_E.solve(**solve_args)
            self.form_constraints()
        else:
            self.active = []

    def form_constraints(self):
        """
        After having fit lasso, form the constraints
        necessary for inference.

        This sets the attributes: `active_constraints`,
        `inactive_constraints`, `active`.

        Returns
        -------

        None

        """

        # determine equicorrelation set and signs
        X, y = self.X, self. y
        beta = self.soln
        n, p = X.shape
        sparsity = s = self.active.shape[0]

        self.z_E = np.sign(beta[self.active])
        X_E = self.X[:,self.active]

        self._linpred_unpenalized = np.dot(X_E, self._beta_unpenalized)
        self._fitted_unpenalized = p_unpen = \
            self.inverse_link(self._linpred_unpenalized)
        self._resid_unpenalized = self.y - self._fitted_unpenalized
        self._weight_unpenalized = self.variance_function(p_unpen)

        Q_Ei = self.quadratic_form_inv # plugin estimate of Q_E(\beta_E^*)^{-1}
        I_E = self._irrepresentable # plugin estimate of C_E(\beta_E^*) Q_E(\beta_E^*)^{-1}

        linear_part = np.zeros((s + 2*(p-s), 2*p))
        linear_part[:s,:s] = -np.diag(self.z_E) 
        linear_part[:s,p+self.active] = self.z_E[:,None] * Q_Ei
        linear_part[s:p,s:p] = np.identity(p-s)
        linear_part[s:p,p+self.active] = I_E
        linear_part[s:p,p+self.inactive] = -np.identity(p-s)

        linear_part[p:] = -linear_part[s:p]

        offset = np.hstack([-self.z_E * np.dot(Q_Ei, self.z_E * self.weights[self.active]),
                             self.weights[self.inactive] - np.dot(I_E, self.z_E * self.weights[self.active]),
                             self.weights[self.inactive] + np.dot(I_E, self.z_E * self.weights[self.active])])

        self._constraints = constraints(linear_part,
                                        offset,
                                        covariance=self.covariance)

        DEBUG = False
        if DEBUG: 
            # check KKT

            # compare to our linearized estimate of inactive gradient

            grad = self.loss.smooth_objective(self.soln, mode='grad') + self.random_linear_term
            print grad[self.active] + self.z_E * self.weights[self.active] , 'should be about 0'
            print np.fabs(grad[self.inactive] / self.weights[self.inactive]).max(), 'should be less than 1'

            approx_grad = (- np.dot(self.X[:,self.inactive].T, self._resid_unpenalized) + self.random_linear_term[self.inactive] 
                           - np.dot(I_E, self.z_E * self.weights[self.active] + self.random_linear_term[self.active]))

            import matplotlib.pyplot as plt
            plt.clf()
            plt.scatter(grad[self.inactive], approx_grad)
            print (np.linalg.norm(grad[self.inactive] - approx_grad) / np.linalg.norm(grad[self.inactive]))

           # not always feasible! 
            print np.linalg.norm(self._feasible[p+self.active] - self.random_linear_term[self.active]) / np.linalg.norm(self.random_linear_term[self.active])

            print (np.dot(self._constraints.linear_part[:s], self._feasible) - (-self.z_E * (self._beta_unpenalized - np.dot(Q_Ei , self.random_linear_term[self.active])))), 'check1'
            print (np.dot(self._constraints.linear_part[:s], self._feasible) - offset[:s]).max(), 'check2'
            print (np.dot(self._constraints.linear_part[s:p], self._feasible) - offset[s:p]), 'check3'

        self.statistical_func = np.hstack([self._beta_unpenalized,
                                           np.dot(self.X[:,self.inactive].T, self._resid_unpenalized),
                                           self.random_linear_term])

        self.feasible = np.zeros(2*p)
        self.feasible[p+self.active] = - self.z_E * self.weights[self.active]
        self.feasible[:s] = np.fabs(self._beta_unpenalized) * self.z_E
        self.feasible[s:p] = np.dot(self.X[:,self.inactive].T, self._resid_unpenalized)
        self.feasible[p+self.inactive] = self.feasible[s:p] + (np.random.sample(self.inactive.shape) * 2 * self.weights[self.inactive] - self.weights[self.inactive])

        if not self._constraints(self.feasible):
            warnings.warn('initial point not feasible for constraints')

    def inverse_link(self, eta):
        return np.exp(eta) / (1 + np.exp(eta))

    def variance_function(self, pi):
        return pi * (1 - pi)

    @property
    def covariance(self, doc="Estimate of covariance of statistical functional."):
        if not hasattr(self, "_cov"):

            # nonparametric bootstrap for covariance of X^Ty

            X, y = self.X, self. y
            n, p = X.shape

            use_parametric = True
            if not use_parametric:
                nsample = 2000
                _mean_cum = 0
                _cov_boot = np.zeros((p, p))

                for _ in range(nsample):
                    indices = np.random.choice(n, size=(n,), replace=True)
                    y_star = y[indices]
                    X_star = X[indices]
                    Z_star = np.dot(X_star.T, y_star - 
                                    self.inverse_link(np.dot(X_star[:,self.active], 
                                                             self._beta_unpenalized)))
                    _mean_cum += Z_star
                    _cov_boot += np.multiply.outer(Z_star, Z_star)
                _cov_boot /= nsample
                _mean = _mean_cum / nsample
                _cov_boot -= np.multiply.outer(_mean, _mean)
                cov_XtY = _cov_boot
            else:
                cov_XtY = np.dot(X.T, self._weight_unpenalized[:, None] * X)

            # the noisier unbiased estimate
            # is equal to the original estimate
            # plus a linear function of the randomization

            # this is the negative of that linear function

            Q_Ei = self.quadratic_form_inv # plugin estimate of Q_E(\beta_E^*)^{-1}
            I_E = self._irrepresentable # plugin estimate of C_E(\beta_E^*) Q_E(\beta_E^*)^{-1}

            sparsity = s = self.active.shape[0]

            self._transform = np.zeros((p, p))
            self._transform[:s,self.active] = Q_Ei
            self._transform[s:,self.inactive] = np.identity(p-s)
            self._transform[s:,self.active] = -I_E

            # now form covariance 

            self._cov = np.zeros((2*p,2*p))
            self._cov[:p,:p] = np.dot(self._transform, np.dot(cov_XtY, self._transform.T))
            self._cov[p:,p:] = self.randomization_covariance

        return self._cov

    @property
    def unbiased_estimate(self, doc=("Selectively unbiased estimate of first half of" + 
                                     " statistical functional.")):
        if not hasattr(self, "_unbiased_estimate"):
            n, p = self.X.shape
            _selective_unbiased_estimate = 0
            linear_func = np.zeros((p, 2*p))
            linear_func[:,:p] = np.identity(p)
            conditional_law = self.constraints.conditional(linear_func,
                                                           self.statistical_func[:p])
            
            burnin = 10000
            ndraw = 30000

            sample = sample_from_constraints(conditional_law,
                                             self.feasible,
                                             self.feasible,
                                             burnin=burnin,
                                             ndraw=ndraw)[:,p:]

            C = self.covariance[:p,:p]
            Cr = self.randomization_covariance
            L = -self._transform

            resid_forming_matrix = R = \
                np.dot(C, 
                        np.linalg.pinv(C + np.dot(L, np.dot(Cr, L.T))))
            self._debiasing_matrix = np.linalg.pinv(np.identity(p) - R)

            self._unbiased_estimate = (self.statistical_func[:p] -
                                       np.mean(np.dot(sample, np.dot(L.T, self._debiasing_matrix.T)), 0))
            self.constraints.mean[:p] = self._unbiased_estimate

        return self._unbiased_estimate

    @property
    def quadratic_form(self, doc="Quadratic form in active block of KKT conditions."):
        if not hasattr(self, "_quad_form"):
            beta = self.soln
            X_E = self.X[:,self.active]
            X_notE = self.X[:,self.inactive]
            self._quad_form = np.dot(X_E.T * self._weight_unpenalized[None, :], X_E)
            self._quad_form_inv = np.linalg.pinv(self._quad_form)
            self._irrepresentable = np.dot(X_notE.T, self._weight_unpenalized[:, None] * np.dot(X_E, self._quad_form_inv))
        return self._quad_form

    @property
    def quadratic_form_inv(self, doc="Inverse of quadratic form in active block of KKT conditions."):
        if not hasattr(self, "_quad_form_inv"):
            self.quadratic_form
        return self._quad_form_inv

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def constraints(self):
        """
        Affine constraints for this LASSO problem.
        This is `self.active_constraints` stacked with
        `self.inactive_constraints`.
        """
        return self._constraints

    #@property
    def intervals(self):
        """
        Intervals for OLS parameters of active variables
        adjusted for selection.

        """
        raise NotImplementedError
#         if not hasattr(self, "_intervals"):
#             self._intervals = []
#             C = self.active_constraints
#             XEinv = self._XEinv
#             if XEinv is not None:
#                 for i in range(XEinv.shape[0]):
#                     eta = XEinv[i]
#                     _interval = C.interval(eta, self.y,
#                                            alpha=self.alpha,
#                                            UMAU=self.UMAU)
#                     self._intervals.append((self.active[i],
#                                             _interval[0], _interval[1]))
#             self._intervals = np.array(self._intervals, 
#                                        np.dtype([('index', np.int),
#                                                  ('lower', np.float),
#                                                  ('upper', np.float)]))
#         return self._intervals

    def hypothesis_test(self, linear_func, null_value=0,
                        alternative='twosided',
                        saturated=True,
                        compute_interval=True,
                        ndraw=8000,
                        burnin=2000):
        """
        Test a null hypothesis of the form

        .. math::

             H_0: \eta^T\beta_E = 0

        in the selected model.

        Parameters
        ----------

        linear_func : np.float(*)
              Linear functional of shape self.active.shape

        null_value : float
              Specified value under null hypothesis

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        saturated : bool
            Should we assume that expected value
            of inactive gradient is 0 or not? If True,
            the test conditions on the asymptotically
            Gaussian $X_{-E}^T(y - \pi_E(\bar{\beta}_E))$.
            One "selected" model assumes this has mean 0.
            The saturated model does not make this assumption.

        compute_interval : bool
             Compute a selective interval?

        ndraw : int (optional)
            Defaults to 8000.

        burnin : int (optional)
            Defaults to 2000.

        Returns
        -------

        pvalue : float
              P-value based on specified alternative.

        interval : (float, float)
              If compute_interval, returns a selective
              interval for that linear functional of the data.
              Otherwise, returns (np.nan, np.nan).

        """

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        n, p = self.X.shape
        sparsity = s = self.active.shape[0]
        null_basis = np.random.standard_normal((s-1,s))
        normsq_linear_func = (linear_func**2).sum()
        for j in range(s-1):
            null_basis[j] -= (null_basis[j] * linear_func) * linear_func / normsq_linear_func

        if saturated:
            conditioning_func = np.zeros((p-1, 2*p))
            conditioning_func[:s-1,:s] = null_basis
            conditioning_func[(s-1):,s:p] = np.identity(p-s)
        else:
            conditioning_func = np.zeros((s-1, 2*p))
            conditioning_func[:s-1,:s] = null_basis

        expanded_linear_func = np.zeros(2*p)
        expanded_linear_func[:s] = linear_func

        conditioning_func = np.dot(conditioning_func, np.linalg.pinv(self.covariance))

        conditional_law = self.constraints.conditional(conditioning_func,
                                                       np.dot(conditioning_func, self.statistical_func))
        if not self.constraints(self.feasible):
            raise ValueError('need a feasible point for sampling')

        sample = sample_from_constraints(conditional_law,
                                         self.feasible,
                                         expanded_linear_func,
                                         ndraw=ndraw,
                                         burnin=burnin)

        W = np.exp(-np.dot(sample, np.dot(np.linalg.pinv(self.constraints.covariance),
                                             self.constraints.mean)))
        family = discrete_family(np.dot(sample, expanded_linear_func),
                                 W)

        observed = (expanded_linear_func * self.statistical_func).sum()

        if compute_interval:
            eta = expanded_linear_func
            lower_lim, upper_lim = family.equal_tailed_interval(observed)
            lower_lim_final = np.dot(eta, np.dot(self.covariance, eta)) * lower_lim
            upper_lim_final = np.dot(eta, np.dot(self.covariance, eta)) * upper_lim

            L_nominal = observed - 1.96 * np.sqrt((eta * np.dot(self.covariance, eta)).sum())
            U_nominal = observed + 1.96 * np.sqrt((eta * np.dot(self.covariance, eta)).sum())
            interval = (lower_lim_final, upper_lim_final)
            print interval, (L_nominal, U_nominal)
        else:
            interval = (np.nan, np.nan)

        if alternative == 'greater':
            pval = family.ccdf(null_value, observed)
        elif alternative == 'less':
            pval = family.cdf(null_value, observed)
        else:
            pval = family.cdf(null_value, observed)
            pval = 2 * min(pval, 1 - pval)
        return pval, interval

def instance(n=100, p=200, s=7, rho=0.3, snr=7,
             random_signs=False, df=np.inf,
             scale=True, center=True):
    """
    A testing instance for the LASSO.
    Design is equi-correlated in the population,
    normalized to have columns of norm 1.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    rho : float
        Equicorrelation value (must be in interval [0,1])

    snr : float
        Size of each coefficient

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    """

    X = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
        np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    if center:
        X -= X.mean(0)[None,:]
    if scale:
        X /= X.std(0)[None,:]
    X /= np.sqrt(n)
    beta = np.zeros(p) 
    beta[:s] = snr 
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True

    # noise model

    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df,size=50000))
            return tdist.rvs(df, size=n) / sd_t

    eta = linpred = np.dot(X, beta) 
    pi = np.exp(eta) / (1 + np.exp(eta))
    Y = np.random.binomial(1, pi)
    return X, Y, beta, np.nonzero(active)[0]


if __name__ == "__main__":
    # from selection.algorithms.lasso import instance

    def simulate(n=400, p=20, burnin=2000, ndraw=8000,
                 compute_interval=False):

        X, y, beta, active = instance(n=n,p=p,snr=8, scale=False, center=False)
        n, p = X.shape

        ybin = y 
        #ybin = np.random.binomial(1, 0.5, size=(n,))
        L = lasso(ybin, X, 0.02 * np.sqrt(n), 
                  randomization_covariance = 0.4 * np.diag(np.diag(np.dot(X.T, X))) / 4.)
        L.fit()
        if (set(range(7)).issubset(L.active) and 
            L.active.shape[0] > 7):
            # L.constraints.mean[:p] = 0 * L.unbiased_estimate
            
            v = np.zeros_like(L.active)
            v[7] = 1.
            P0, interval = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                             compute_interval=compute_interval,
                                             saturated=False)
            target = (beta[L.active]*v).sum()
            estimate = (L.unbiased_estimate[:L.active.shape[0]]*v).sum()
            low, hi = interval

            v = np.zeros_like(L.active)
            v[0] = 1.
            PA, _ = L.hypothesis_test(v, burnin=burnin, ndraw=ndraw,
                                      compute_interval=compute_interval,
                                      saturated=False)

            return P0, PA, L
