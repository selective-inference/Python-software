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
from warnings import warn

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

def draw_truncated(initial, C, nstep=10):
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
    final = initial.copy()
    n = final.shape[0]
    for _ in range(nstep):
      final = gibbs_step(np.random.standard_normal(n), final, C)
    return final

def expected_norm_squared(initial, C, ndraw=300):
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

    Returns
    -------

    expected_value : `np.float`
        An estimate of the expected Euclidean norm squared 
        under the distribution implied by $C$.

    variance : `np.float`
        An estimate of the varaince of the Euclidean norm squared 
        under the distribution implied by $C$.

    """
    sample = []
    state = initial.copy()
    state = draw_truncated(initial, C)
    for _ in range(ndraw):
        state = draw_truncated(state, C)
        sample.append(state.copy())
    return np.mean(np.sum(np.array(sample)**2, 1)), np.std(np.sum(np.array(sample)**2, 1))**2

def estimate_sigma(Y, C, niter=500):
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

    Returns
    -------

    S_hat : float
        An estimate of $\sigma$.

    """

    n = Y.shape[0]
    observed = (Y**2).sum()
    initial = np.sqrt(observed / n)

    S_hat = initial
    
    for i in range(niter):
        C.covariance = S_hat**2 * np.identity(n)
        E = expected_norm_squared(Y, C, ndraw=20)[0]
        grad = (observed - E) / S_hat**3
        step = grad / (i + 1)
        
        S_trial = S_hat + step
        while S_trial < 0:
            step /= 2
            S_trial = S_hat + step
        S_hat = S_trial
    return S_hat
        
def estimate_sigma_newton(Y, C, niter=40, ndraw=500):
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

    Returns
    -------

    S_hat : float
        An estimate of $\sigma$.

    """
    n = Y.shape[0]
    observed = (Y**2).sum()
    initial = np.sqrt(observed / n)

    S = initial
    G = -1./S**2
    for i in range(niter):
        C.covariance = S**2 * np.identity(n)
        E, V = expected_norm_squared(Y, C, ndraw=1000)
        grad = (observed - E) / 2
        hessian = V / 4
        step = (grad / hessian) 

        G_trial = G + step
        while G_trial > 0:
            step /= 2
            G_trial = G + step
        G = G_trial
        S = 1. / np.sqrt(-G)

    S_hat = S
    return S_hat
