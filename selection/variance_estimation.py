"""
This module implements a conditional MLE
for $\sigma$ when a constraint `con` is assumed to have
`con.covariance` equal to $\sigma^2 I_{n \times n}$ with
$n$ being `con.dim`.

It is based on a simple Gibbs scheme to sample from a 
Gaussian with this covariance constrained to lie in $C$ where
$C$ is the region specified by the inequality constraints in
`con`.  

Constraints with equality constraints are not handled by this 
algorithm but could be handled by a simple modification of the Gibbs scheme.
"""

import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from scipy.integrate import quad

from warnings import warn

#from sklearn.isotonic import IsotonicRegression

# load rpy2 and initialize for numpy support
import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
numpy2ri.activate()

from mpmath import quad as mpquad, exp as mpexp, log as mplog, mp
mp.dps = 60

from .chisq import quadratic_constraints

DEBUG = False

def gibbs_step(direction, Y, C):
    """
    Take a Gibbs step in given direction from $Y$
    where we assume $Y$ is a realization of
    a Gaussian with covariance `C.covariance`
    (assumed to be a multiple of the identity)
    truncated to the inequality constraint specified
    by `C`.

    Parameters
    ----------

    direction : `np.float`
        Direction in which to take the step.

    Y : `np.float`
        Realization of the random vector. 
        Should satisfy the constraints of $C$.

    C : `constraints`
        Constraints which will be satisfied after
        taking the Gibbs step.

    """

    if not C(Y): # check whether Y is in the cone
        warn('Y does not satisfy the constraints')

    direction = direction / np.linalg.norm(direction)
    L, _, U, S = C.bounds(direction, Y)
    L_std, U_std = L/S, U/S # standardize the endpoints
    
    trunc = truncnorm(L_std, U_std)
    sample = trunc.rvs()  
    
    # the sampling sometimes returns np.inf in this case, we use 
    # an exponential approximation
    
    if sample == np.inf:
        sample = L_std + np.random.exponential(1) / L_std
    elif sample == -np.inf:
        sample = U_std - np.random.exponential(1) / U_std
        
    # now take the step
    Y_perp = Y - (Y*direction).sum() / (direction**2).sum() * direction
    return Y_perp + sample * direction * S # reintroduce the scale 

def draw_truncated(initial, C, ndraw=1000, burnin=1000):
    """
    Starting with a point in $C$, simulate by taking `nstep` 
    Gibbs steps and return the resulting point.

    Parameters
    ----------

    initial : `np.float`
        State at which to begin Gibbs steps from.

    C : `constraints`
        Constraints which will be satisfied after
        taking the Gibbs step.

    Returns
    -------

    final : `np.float`
        State after taking a certain number of Gibbs steps.

    """
    state = initial.copy()
    n = state.shape[0]
    sample = np.zeros((ndraw,n))
    for i in range(ndraw + burnin):
      state = gibbs_step(np.random.standard_normal(n), state, C)
      if i >= burnin:
          sample[i-burnin] = state.copy()
    return sample

def expected_norm_squared(initial, C, ndraw=1000, burnin=1000):
    """
    Starting with a point in $C$, estimate
    the expected value of $\|Y\|^2_2$ under
    the distribution implied by $C$, as well
    as the variance.

    Parameters
    ----------

    initial : `np.float`
        State at which to begin Gibbs steps from.

    C : `constraints`
        Constraints which will be satisfied after
        taking the Gibbs step.

    ndraw : int
        How many draws should be used to 
        estimate the mean and variance.

    burnin : int
        How many iterations for burnin.

    Returns
    -------

    expected_value : `np.float`
        An estimate of the expected Euclidean norm squared 
        under the distribution implied by $C$.

    variance : `np.float`
        An estimate of the varaince of the Euclidean norm squared 
        under the distribution implied by $C$.

    """

    sample = draw_truncated(initial, C, ndraw=ndraw, burnin=burnin)
    return np.mean(np.sum(np.array(sample)**2, 1)), np.std(np.sum(np.array(sample)**2, 1))**2, state

def estimate_sigma(Y, C, niter=100, ndraw=100, inflation=4):
    """
    A stochastic gradient descent algorithm
    to estimate $\sigma$ assuming that
    $Y$ is a realization of the distribution
    implied by $C$ (assumes that `C.mean == 0`).

    Parameters
    ----------

    Y : `np.float`
        Realization to use to estimate $\sigma$

    C : `constraints`
        Constraints which will be satisfied after
        taking the Gibbs step.

    niter : int
        How many iterations of the
        stochastic optimization algorithm
        should we use.

    ndraw : int
        How many draws should we use to form estimate
        the gradient.

    inflation : float
        Factor to inflate naive estimate of $\sigma$ by as an 
        initial condition.
    
    Returns
    -------

    S_hat : float
        An estimate of $\sigma$.

    """

    n = Y.shape[0]
    observed = (Y**2).sum()
    initial = inflation * np.sqrt(observed / n)

    S_hat = initial
    state = Y.copy()

    for i in range(niter):
        C.covariance = S_hat**2 * np.identity(n)
        E, _, state = expected_norm_squared(state, C, ndraw=ndraw)
        grad = (observed - E) / S_hat**3
        step = grad / (i + 1)**(0.75)
        
        S_trial = S_hat + step
        while S_trial < 0:
            step /= 2
            S_trial = S_hat + step
        S_hat = S_trial
        if DEBUG:
            print S_hat
    return S_hat
        
def estimate_sigma_newton(Y, C, niter=40, ndraw=500, inflation=4):
    """
    A quasi-Newton algorithm
    to estimate $\sigma$ assuming that
    $Y$ is a realization of the distribution
    implied by $C$ (assumes that `C.mean == 0`).

    Parameters
    ----------

    Y : `np.float`
        Realization to use to estimate $\sigma$

    C : `constraints`
        Constraints which will be satisfied after
        taking the Gibbs step.

    niter : int
        How many iterations of the
        stochastic optimization algorithm
        should we use.

    ndraw : int
        How many draws should we use to form estimate
        the gradient and Hessian.

    inflation : float
        Factor to inflate naive estimate of $\sigma$ by as an 
        initial condition.
    
    Returns
    -------

    S_hat : float
        An estimate of $\sigma$.

    """
    n = Y.shape[0]
    observed = (Y**2).sum()
    initial = inflation * np.sqrt(observed / n)

    S = initial
    G = -1./S**2
    state = Y.copy()

    alpha = initial / 10.

    for i in range(niter):
        C.covariance = S**2 * np.identity(n)
        E, V, state = expected_norm_squared(Y, C, ndraw=ndraw)
        
        step = alpha * np.sign(observed - E) 
        S_trial = S + step
        C.covariance = S_trial**2 * np.identity(n)
        new_E = expected_norm_squared(Y, C, ndraw=ndraw)[0]

        while np.sign(observed - E) != np.sign(observed - new_E):
            step /= 2
            S_trial = S + step
            C.covariance = S_trial**2 * np.identity(n)
            new_E = expected_norm_squared(Y, C, ndraw=ndraw)[0]
            if DEBUG:
                print (S, S_trial, np.sign(observed - E), np.sign(observed - new_E), observed, E, new_E)
            
        #G = G_trial
        S = S_trial
        if DEBUG:
            print S, observed, E, new_E

    S_hat = S
    return S_hat

def interpolation_estimate(Z, Z_constraint,
                           lower=0.5,
                           upper=4,
                           npts=30,
                           ndraw=5000,
                           burnin=1000,
                           estimator='truncated'):
    """
    Estimate the parameter $\sigma$ in $Z \sim N(0, \sigma^2 I) | Z \in C$
    where $C$ is the convex set encoded by `Z_constraints`

    .. math::

       C = \left\{z: Az+b \geq 0 \right\}

    with $(A,b)$ being `(Z_constraints.inequality, 
    Z_constraints.inequality_offset)`.

    The algorithm proceeds by estimating $\|Z\|^2_2$ 
    by Monte Carlo for a range of `npts` values starting from
    `lower*np.linalg.norm(Z)/np.sqrt(n)` to
    `upper*np.linalg.norm(Z)/np.sqrt(n)` with `n=Z.shape[0]`.

    These values are then used to compute the GCM 
    (Greated Convex Minorant) which is interpolated and solved 
    for an arguments such that the expected value matches the observed
    value `(Z**2).sum()`.

    Parameters
    ----------

    Z : `np.float`
        Observed data to be used to estimate $\sigma$. Should be in
        the cone specified by `Z_constraints`.

    Z_constraint : `constraints`
        Constraints under which we observe $Z$.

    lower : float
        Multiple of naive estimate to use as lower endpoint.

    upper : float
        Multiple of naive estimate to use as upper endpoint.

    npts : int
        Number of points in interpolation grid.

    ndraw : int
        Number of Gibbs steps to use for estimating
        each expectation.

    burnin : int
        How many Gibbs steps to use for burning in.

    Returns
    -------

    sigma_hat : float
        The root of the interpolant derived from GCM values.

    interpolant : `interp1d`
        The interpolant, to be used for plotting or other 
        diagnostics.

    WARNING
    -------

    * It is assumed that `Z_constraints.equality` is `None`.
    
    * Uses `rpy2` and `fdrtool` library to compute the GCM.

    """

    initial = np.linalg.norm(Z) / np.sqrt(Z.shape[0])

    Svalues = np.linspace(lower*initial,upper*initial, npts)
    Evalues = []

    n = Z.shape[0]
    L, V, U, S = quadratic_constraints(Z, np.identity(n), Z_constraint)

    if estimator == 'truncated':
        def _estimator(S, Z, Z_constraint):
            L, V, U, _ = quadratic_constraints(Z, np.identity(n), Z_constraint)
            num = mpquad(lambda x: mpexp(-x**2/(2*S**2) -L*x / S**2 + (n-1) * mplog((x+L)/S) + 2 * mplog(x+L)),
                       [0, U-L])
            den = mpquad(lambda x: mpexp(-x**2/(2*S**2) -L*x / S**2 + (n-1) * mplog((x+L)/S)),
                       [0, U-L])
            print num / den, V**2, S, (L, U)
            return num / den
    elif estimator == 'simulate':
        
        state = Z.copy()
        rpy.r.assign('state', state)
        def _estimator(S, state, Z_constraint):
            Z_constraint.covariance = S**2 * np.identity(Z.shape[0])
            e, v, _state = expected_norm_squared(state, 
                                               Z_constraint, ndraw=ndraw,
                                               burnin=burnin)            
            state[:] = _state
            return e

    state = Z.copy()
    for S in Svalues:
        Evalues.append(_estimator(S, state, Z_constraint))
    ir = IsotonicRegression()
    if DEBUG:
        print Svalues, Evalues
    Eiso = ir.fit_transform(Svalues, Evalues)
    Sinterp, Einterp = Svalues, Eiso
#     rpy.r.assign('S', Svalues)
#     rpy.r.assign('E', np.array(Evalues))
#     rpy.r('''
#     library(fdrtool);
#     G = gcmlcm(S, E, 'gcm');
#     Sgcm = G$x.knots;
#     Egcm = G$y.knots;
#     ''')
#     Sgcm = np.asarray(rpy.r('Sgcm'))
#     Egcm = np.asarray(rpy.r('Egcm'))
#     interpolant = interp1d(Sgcm, Egcm - (Z**2).sum())

    interpolant = interp1d(Sinterp, Einterp - (Z**2).sum())
    try:
        sigma_hat = bisect(interpolant, Sinterp.min(), Sinterp.max())
    except:
        raise ValueError('''Bisection failed -- check (lower, upper). Observed = %0.1e, Range = (%0.1e,%0.1e)''' % ((Z**2).sum(), Einterp.min(), Einterp.max()))
    return sigma_hat, interpolant

def truncated_estimate(Z, Z_constraint,
                      lower=0.5,
                      upper=2,
                      npts=15):
    """
    Estimate the parameter $\sigma$ in $Z \sim N(0, \sigma^2 I) | Z \in C$
    where $C$ is the convex set encoded by `Z_constraints`

    .. math::

       C = \left\{z: Az+b \geq 0 \right\}

    with $(A,b)$ being `(Z_constraints.inequality, 
    Z_constraints.inequality_offset)`.

    The algorithm proceeds by estimating $\|Z\|^2_2$ 
    by Monte Carlo for a range of `npts` values starting from
    `lower*np.linalg.norm(Z)/np.sqrt(n)` to
    `upper*np.linalg.norm(Z)/np.sqrt(n)` with `n=Z.shape[0]`.

    These values are then used to compute the GCM 
    (Greated Convex Minorant) which is interpolated and solved 
    for an arguments such that the expected value matches the observed
    value `(Z**2).sum()`.

    Parameters
    ----------

    Z : `np.float`
        Observed data to be used to estimate $\sigma$. Should be in
        the cone specified by `Z_constraints`.

    Z_constraint : `constraints`
        Constraints under which we observe $Z$.

    lower : float
        Multiple of naive estimate to use as lower endpoint.

    upper : float
        Multiple of naive estimate to use as upper endpoint.

    npts : int
        Number of points in interpolation grid.

    Returns
    -------

    sigma_hat : float
        The root of the interpolant derived from GCM values.

    interpolant : `interp1d`
        The interpolant, to be used for plotting or other 
        diagnostics.

    WARNING
    -------

    * It is assumed that `Z_constraints.equality` is `None`.
    
    * Uses `rpy2` and `fdrtool` library to compute the GCM.

    """

    initial = np.linalg.norm(Z) / np.sqrt(Z.shape[0])

    Svalues = np.linspace(lower*initial,upper*initial, npts)
    Evalues = []

    # use truncated chi to estimate integral
    # with scipy.integrate.quad
    n = Z.shape[0]
    operator = np.identity(n)
    L, V, U, S = quadratic_constraints(Z, operator, Z_constraint)

    for S in Svalues:
        num = quad(lambda x: np.exp(-x**2/(2*S**2) + (n+1) * np.log(x)),
                   L, U)
        den = quad(lambda x: np.exp(-x**2/(2*S**2) + (n-1) * np.log(x)),
                   L, U)
        Evalues.append(num[0] / den[0])
        print num, den

    ir = IsotonicRegression()
    if DEBUG:
        print Svalues, Evalues
    Eiso = ir.fit_transform(Svalues, Evalues)
    Sinterp, Einterp = Svalues, Eiso


    interpolant = interp1d(Sinterp, Einterp - (Z**2).sum())
    try:
        sigma_hat = bisect(interpolant, Sinterp.min(), Sinterp.max())
    except:
        raise ValueError('''Bisection failed -- check (lower, upper). Observed = %0.1e, Range = (%0.1e,%0.1e)''' % ((Z**2).sum(), Einterp.min(), Einterp.max()))
    return sigma_hat, interpolant


    print L, V, U, S

