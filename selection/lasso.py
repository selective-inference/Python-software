import numpy as np
import regreg.api as rr
from .constraints import constraints, selection_interval
from .intervals import pivot
from scipy.stats import norm as ndist
import warnings

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('cvx not available')
    pass

DEBUG = False

class FixedLambdaError(ValueError):
    pass

def interval_constraint_linf(Z, S, offset, 
                             lower_bound=None,
                             upper_bound=None):
    '''

    Compute the maximum of np.fabs(Z) and return 
    an interval within which it is constrained to lie. 
    The interval is constructed so that, if Z has
    covariance matrix S, then the upper and lower
    end points are independent of Z[j]
    conditional on (j,s) = (j_star, s_star)
    where s_star * Z[j_star] = np.max(np.fabs(Z)).

    This is used for a p-value under the assumption 
    $y \sim N(\mu, \sigma^2 I)$.

    Parameters
    ==========

    Z : np.array((p,))
        Response vector.

    S : np.array((p,p))
        Covariance matrix

    offset : np.array((p,))
        Hypothesized offset of response vector.

    lower_bound : np.array((p,)) (optional)
        A vector of lower bound that X^Ty is constrained to lie above.

    upper_bound : np.array((p,)) (optional)
        A vector of upper bounds that X^Ty is constrained to lie below.

    Returns
    =======

    L : np.float
        Maximum of np.fabs(np.dot(X.T,y)),
        possibly after having added offset to y.

    Vplus : np.float
        A lower bound for L.

    Vminus : np.float
        An upper bound for L.

    var_star : np.float
        Variance of np.dot(X.T,y) evaluated at argmax.

    offset_star : 
        Offset vector evaluated at argmax, multiplied by sign at
        argmax.

    '''

    Z += offset
    j_star = np.argmax(np.fabs(Z))
    s_star = np.sign(Z[j_star])
    offset_star = offset[j_star] * s_star
    Z_star = Z[j_star]
    
    L = np.fabs(Z).max()
    var_star = S[j_star, j_star]
    C_X = s_star * S[j_star] / var_star

    Mplus = {}
    Mminus = {}
    keep = np.ones(Z.shape[0], np.bool)
    keep[j_star] = 0
    
    den = 1 - C_X
    num = Z - C_X * L
    Mplus[1] = (num / den * (den > 0))[keep]
    Mminus[1] = (num * keep / (den + (1 - keep)))
    
    den = 1 + C_X
    num =  -(Z - C_X * L)
    Mplus[-1] = (num / den * (den > 0))[keep]
    Mminus[-1] = (num * keep / (den + (1 - keep)))[den < 0]
    
    mplus = np.hstack([Mplus[1],Mplus[-1]])
    Vplus = np.max(mplus)
    
    mminus = []
    if Mminus[1].shape:
        mminus.extend(list(Mminus[1]))
    if Mminus[-1].shape:
        mminus.extend(list(Mminus[-1]))
    if mminus:
        mminus = np.array(mminus)
        mminus = mminus[mminus > L]
        if mminus.shape != (0,):
            Vminus = mminus.min()
        else:
            Vminus = np.inf
    else:
        Vminus = np.inf
    
    # enforce the interval constraint

    if DEBUG:
        print 'before:', Vplus, L, Vminus

    if upper_bound is not None:
        # we need to rewrite all constraints
        # as an inequality between Z[j_star] and 
        # something independent of Z[j_star]

        u_star = upper_bound[j_star]

        W = (upper_bound - (Z - L * C_X)) / C_X

        pos_coords = (C_X > 0) * keep
        if np.any(pos_coords): 
            pos_implied_bounds = W[pos_coords]
            Vminus = min(Vminus, pos_implied_bounds.min())

        neg_coords = (C_X < 0) * keep

        if np.any(neg_coords):
            neg_implied_bounds = W[neg_coords]
            Vplus = max(Vplus, neg_implied_bounds.max())

        if s_star == 1:
            Vminus = min(Vminus, s_star * u_star)
        else:
            Vplus = max(Vplus, s_star * u_star)

    if DEBUG:
        print 'upper:', Vplus, L, Vminus

    if lower_bound is not None:

        l_star = lower_bound[j_star]

        W = (lower_bound - (Z - L * C_X)) / C_X

        pos_coords = (C_X > 0) * keep
        if np.any(pos_coords): 
            pos_implied_bounds = W[pos_coords]
            Vplus = max(Vplus, pos_implied_bounds.max())

        neg_coords = (C_X < 0) * keep

        if np.any(neg_coords):
            neg_implied_bounds = W[neg_coords]
            Vminus = min(Vminus, neg_implied_bounds.min())

        if s_star == 1:
            Vplus = max(Vplus, s_star * l_star)
        else:
            Vminus = min(Vminus, s_star * l_star)

    if DEBUG:
        print 'lower:', Vplus, L, Vminus

    return L, Vplus, Vminus, var_star, offset_star
 
def fixed_pvalue_uncentered(y, X, lagrange, soln, sigma_epsilon=1):
    """
    Compute a p-value for testing whether the LASSO
    has found all important variables at some fixed
    percentage of $\lambda_1 = \|X^Ty\|_{\infty}$
    under model $y \sim N(X\beta_0, \sigma^2 I)$.
    
    This test is based on the uncentered inactive subgradient
    
    .. math::

       X_{-E}^T((I-P_E)y + \lambda (X_E^T)^{\dagger} s_E)

    which is constrained to have $\ell_{\infty}$ norm
    less than `lagrange` by the KKT conditions.

    Parameters
    ==========

    y : np.array(n)
        Response vector.

    X : np.array((n,p))
        Design matrix

    lagrange : float
        How far down the regularization path should we test?

    soln : np.array(p)
        Solution at this value of L

    sigma_epsilon : float
        Standard deviation of noise, $\sigma$.

    """

    n, p = X.shape

    nonzero_coef = soln != 0
    tight_subgrad = np.fabs(np.fabs(np.dot(X.T, y - np.dot(X, soln))) / lagrange - 1) < 1.e-3
    if DEBUG:
        print 'KKT consistency', (nonzero_coef - tight_subgrad).sum()

    A = nonzero_coef

    if A.sum() > 0:
        sA = np.sign(soln[A])
        XA = X[:,A]
        XnotA = X[:,~A]
        XAinv = np.linalg.pinv(XA)
        PA = np.dot(XA, XAinv)
        irrep_subgrad = lagrange * np.dot(np.dot(XnotA.T, XAinv.T), sA)

    else:
        XnotA = X
        PA = 0
        irrep_supgrad = np.zeros(p)

    if A.sum() < X.shape[1]:
        inactiveX = np.dot(np.identity(n) - PA, XnotA)
        scaling = np.sqrt((inactiveX**2).sum(0))
        inactiveX /= scaling[None,:]
        upper_bound = lagrange * np.zeros(inactiveX.shape[1])
        lower_bound = -upper_bound
        covX = np.dot(inactiveX.T, inactiveX)

        L, Vp, Vm, var_star, offset_star = \
            interval_constraint_linf(np.dot(inactiveX.T, y), covX, 
                                     irrep_subgrad,
                                     lower_bound=lower_bound,
                                     upper_bound=upper_bound)
        sigma = np.sqrt(var_star) * sigma_epsilon

        if np.isnan(Vp):
            raise FixedLambdaError('saturated solution?')

        pval = pivot(Vp-offset_star, L-offset_star, Vm-offset_star, sigma, dps=30)
        return pval, soln, (Vp-offset_star) / sigma, (L-offset_star) / sigma, (Vm-offset_star) / sigma
    else:
        pval = 1.
        soln = soln
        return np.clip(pval, 0, 1), soln, None, None, None

def fixed_pvalue_centered(y, X, lagrange, soln, sigma_epsilon=1):
    """
    Compute a p-value for testing whether the LASSO
    has found all important variables at some fixed
    percentage of $\lambda_1 = \|X^Ty\|_{\infty}$
    under model $y \sim N(X\beta_0, \sigma^2 I)$.
    
    This test is based on a centered, scaled
    inactive subgradient
    
    .. math::

       X_{-E}^T(I-P_E)y

    subject to the constraints imposed by the KKT conditions.
    That is, it is contained within
    an $\ell_{\infty}$ ball of radius `lagrange`
    of $-X_{-E}^T(X_E^T)^{\dagger}s_e$.

    The scaling is such that each entry above has constant
    variance.

    Parameters
    ==========

    y : np.array(n)
        Response vector.

    X : np.array((n,p))
        Design matrix

    lagrange : float
        How far down the regularization path should we test?

    soln : np.array(p)
        Solution at this value of L

    sigma_epsilon : float
        Standard deviation of noise, $\sigma$.

    """

    n, p = X.shape

    nonzero_coef = soln != 0
    tight_subgrad = np.fabs(np.fabs(np.dot(X.T, y - np.dot(X, soln))) / lagrange - 1) < 1.e-3
    if DEBUG:
        print 'KKT consistency', (nonzero_coef - tight_subgrad).sum()

    A = nonzero_coef

    if A.sum() > 0:
        sA = np.sign(soln[A])
        XA = X[:,A]
        XnotA = X[:,~A]
        XAinv = np.linalg.pinv(XA)
        PA = np.dot(XA, XAinv)
        irrep_subgrad = lagrange * np.dot(np.dot(XnotA.T, XAinv.T), sA)

    else:
        XnotA = X
        PA = 0
        irrep_supgrad = np.zeros(p)

    if A.sum() < X.shape[1]:
        inactiveX = np.dot(np.identity(n) - PA, XnotA)
        scaling = np.sqrt((inactiveX**2).sum(0))
        inactiveX /= scaling[None,:]
        upper_bound = (lagrange - irrep_subgrad) / scaling
        lower_bound = (- lagrange - irrep_subgrad) / scaling
        covX = np.dot(inactiveX.T, inactiveX)

        L, Vp, Vm, var_star, offset_star = \
            interval_constraint_linf(np.dot(inactiveX.T, y), covX, 
                                     np.zeros(covX.shape[0]),
                                     lower_bound=lower_bound,
                                     upper_bound=upper_bound)
        sigma = np.sqrt(var_star) * sigma_epsilon

        if np.isnan(Vp):
            raise FixedLambdaError

        pval = pivot(Vp, L, Vm, sigma, dps=30)
        return pval, soln, (Vp-offset_star) / sigma, (L-offset_star) / sigma, (Vm-offset_star) / sigma
    else:
        pval = 1.
        soln = soln
        return np.clip(pval, 0, 1), soln, None, None, None

class lasso(object):

    """
    A class for the LASSO that performs the two tests,
    and returns selection intervals for the 
    active coefficients.

    WARNING: Changing `frac` will not propagate:
    constraints will not be updated, not soln, etc.
    
    Better to create a new instance.

    """
    
    # level for coverage is 1-alpha
    alpha = 0.05

    def __init__(self, y, X, frac=0.9, sigma_epsilon=1):
        self.y = y
        self.X = X
        self.frac = frac
        self.sigma_epsilon = sigma_epsilon
        self.lagrange = frac * np.fabs(np.dot(X.T, y)).max()
        self._covariance = self.sigma_epsilon**2 * np.identity(X.shape[0])

    def fit(self, tol=1.e-8,
            min_its=50, use_cvx=False):
        """
        self.soln only updated after self.fit
        """

        if use_cvx: 
            beta = cvx.variable(p)
            _X = cvx.parameter(n,p)
            _X.value = cvx.matrix(X)

            _Y = cvx.parameter(n,1)
            _Y.value = cvx.matrix(y.reshape((n,1)))

            beta = cvx.variable(p)

            objective = cvx.sum(cvx.square(_Y-_X*beta))
            penalty = cvx.sum(cvx.abs(beta))
            program = cvx.program(cvx.minimize(0.5*objective + lagrange*penalty))
            program.solve(quiet=True)

            soln = np.array(beta.value).reshape(-1)
            soln[np.fabs(soln) < tol * np.fabs(soln).max()] = 0
        else: # use regreg
            X, y = self.X, self.y
            n, p = self.X.shape
            penalty = rr.l1norm(p, lagrange=self.lagrange)
            loss = rr.squared_error(X, y)
            problem = rr.simple_problem(loss, penalty)
            soln = problem.solve(tol=tol, min_its=min_its)
        self._soln = soln
        return self._soln

    @property
    def soln(self):
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def centered_test(self):
        return fixed_pvalue_centered(self.y, 
                                     self.X, 
                                     self.lagrange, 
                                     self.soln, 
                                     sigma_epsilon=self.sigma_epsilon)[:2]

    @property
    def basic_test(self):
        return fixed_pvalue_uncentered(self.y, 
                                       self.X, 
                                       self.lagrange, 
                                       self.soln, 
                                       sigma_epsilon=self.sigma_epsilon)[:2]


    @property
    def constraints(self, doc="Constraint matrix for this LASSO problem"):
        if not hasattr(self, "_constraints"):
            X, y, soln, lagrange = self.X, self.y, self.soln, self.lagrange
            n, p = X.shape

            nonzero_coef = soln != 0
            tight_subgrad = np.fabs(np.fabs(np.dot(X.T, y - np.dot(X, soln))) / lagrange - 1) < 1.e-3
            if DEBUG:
                print 'KKT consistency', (nonzero_coef - tight_subgrad).sum()

            A = nonzero_coef
            self.active = np.nonzero(nonzero_coef)[0]
            if A.sum() > 0:
                sA = np.sign(soln[A])
                XA = X[:,A]
                XnotA = X[:,~A]
                self._XAinv = XAinv = np.linalg.pinv(XA)
                active_constraints = [(sA[:,None] * XAinv, np.zeros(sA.shape))]
                PA = np.dot(XA, XAinv)
                irrep_subgrad = lagrange * np.dot(np.dot(XnotA.T, XAinv.T), sA)

            else:
                XnotA = X
                PA = 0
                irrep_supgrad = np.zeros(p)
                active_constraints = []

            if A.sum() < X.shape[1]:
                inactiveX = np.dot(np.identity(n) - PA, XnotA)
                scaling = np.sqrt((inactiveX**2).sum(0))
                inactiveX /= scaling[None,:]

                inactive_constraints = [(-inactiveX.T, (lagrange - 
                                                        irrep_subgrad) / scaling),
                                        (inactiveX.T, (lagrange + 
                                                        irrep_subgrad) / scaling)]
            else:
                inactive_constraints = []

            _constraints = active_constraints + inactive_constraints
            _linear_part = np.vstack([A for A, _ in _constraints])
            _offset_part = np.hstack([b for _, b in _constraints])
            self._constraints = constraints((_linear_part, 
                                             _offset_part),
                                            None)

        return self._constraints

    @property
    def intervals(self, doc="OLS intervals for active variables adjusted for selection."):
        if not hasattr(self, "_intervals"):
            self._intervals = []
            C = self.constraints
            XAinv = self._XAinv
            for i in range(XAinv.shape[0]):
                eta = XAinv[i]
                _interval = selection_interval( \
                       C.inequality,
                       C.inequality_offset,
                       self._covariance,
                       self.y,
                       eta,
                       dps=15,
                       upper_target=1-self.alpha/2,
                       lower_target=self.alpha/2)
                self._intervals.append((self.active[i], eta, (eta*self.y).sum(), 
                                        _interval))
        return self._intervals

    @property
    def unadjusted_intervals(self, doc="Unadjusted OLS intervals for active variables."):
        if not hasattr(self, "_intervals_unadjusted"):
            self.constraints # force _XAinv to be computed -- 
                             # bad use of property
            self._intervals_unadjusted = []
            XAinv = self._XAinv
            SigmaA = np.dot(XAinv, XAinv.T) * self.sigma_epsilon**2
            for i in range(XAinv.shape[0]):
                eta = XAinv[i]
                center = (eta*self.y).sum()
                width = ndist.ppf(1-self.alpha/2.) * np.sqrt(SigmaA[i,i])
                _interval = [center-width, center+width]
                self._intervals_unadjusted.append((self.active[i], eta, (eta*self.y).sum(), 
                                        _interval))
        return self._intervals_unadjusted

def fit_and_test(y, X, frac, sigma_epsilon=1, use_cvx=False,
                 test='centered',
                 tol=1.e-8,
                 min_its=50):
    """
    Fit a LASSO at some fraction of $\lambda_1$, return 
    solution and $p$-value based on solution.

    The tolerance, when use_cvx=True is used to
    zero out coordinates, so coordinates are set to zero if their absolute
    value is less than tol times the $\ell_{\infty}$ norm.
    
    When not using cvx, it used as tol for regreg, as
    are the min_its.

    """
    n, p = X.shape

    lagrange = frac*np.fabs(np.dot(X.T,y)).max()

    if use_cvx: 
        beta = cvx.variable(p)
        _X = cvx.parameter(n,p)
        _X.value = cvx.matrix(X)

        _Y = cvx.parameter(n,1)
        _Y.value = cvx.matrix(y.reshape((n,1)))

        beta = cvx.variable(p)

        objective = cvx.sum(cvx.square(_Y-_X*beta))
        penalty = cvx.sum(cvx.abs(beta))
        program = cvx.program(cvx.minimize(0.5*objective + lagrange*penalty))
        program.solve(quiet=True)

        soln = np.array(beta.value).reshape(-1)
        soln[np.fabs(soln) < tol * np.fabs(soln).max()] = 0
    else:
        penalty = rr.l1norm(p, lagrange=lagrange)
        loss = rr.squared_error(X, y)
        problem = rr.simple_problem(loss, penalty)
        soln = problem.solve(tol=tol, min_its=min_its)
    
    if test == 'centered':
        return fixed_pvalue_centered(y, X, lagrange, soln, sigma_epsilon=sigma_epsilon)
    elif test == 'uncentered':
        return fixed_pvalue_uncentered(y, X, lagrange, soln, sigma_epsilon=sigma_epsilon)
    elif test == 'both':
        v1, v3 = fixed_pvalue_centered(y, X, lagrange, soln, sigma_epsilon=sigma_epsilon)
        v2 = fixed_pvalue_uncentered(y, X, lagrange, soln, sigma_epsilon=sigma_epsilon)
        return v1, v2, v3
    else:
        raise FixedLambdaError('test must be one of ["centered", "uncentered", "both"]')

