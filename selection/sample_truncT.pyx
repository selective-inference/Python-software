import numpy as np, cython
cimport numpy as np

from libc.math cimport pow, sqrt, log, exp # sin, cos, acos, asin, sqrt, fabs
from scipy.special import ndtr, ndtri

cdef double PI = np.pi

"""
This module has a code to sample from a truncated normal distribution
specified by a set of affine constraints.
"""

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncated_T(np.ndarray[DTYPE_float_t, ndim=2] A, 
                       np.ndarray[DTYPE_float_t, ndim=1] b, 
                       np.ndarray[DTYPE_float_t, ndim=1] initial, 
                       np.ndarray[DTYPE_float_t, ndim=1] noncentrality, 
                       int degrees_of_freedom,
                       # eta, the direction of interest
                       np.ndarray[DTYPE_float_t, ndim=1] bias_direction, 
                       DTYPE_int_t how_often=1000,
                       DTYPE_int_t burnin=500,
                       DTYPE_int_t ndraw=1000,
                       DTYPE_int_t discretization=1000,
                       ):
    """
    Sample from a truncated multivariate T with covariance
    equal to I and a given noncentrality parameter and degrees
    of freedom.

    Constraint is $Ax \leq b$ where `A` has shape
    `(q,n)` with `q` the number of constraints and
    `n` the number of random variables.


    Parameters
    ----------

    A : np.float((q,n))
        Linear part of affine constraints.

    b : np.float(q)
        Offset part of affine constraints.

    initial : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    noncentrality : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    bias_direction : np.float
        Which projection is of most interest? 

    degrees_of_freedom : int
        Degrees of freedom of multivariate T.

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    discretization : int
        How many points to discretize truncation interval to.

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

    cdef int df = degrees_of_freedom
    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double cdfL, cdfU, unif, tnorm, val, alpha

    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] U = np.dot(A, state) - b

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes

    _dirs = [np.random.standard_normal((nvar,nvar))]

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack(_dirs)
        
    directions[-1][:] = bias_direction 

    directions /= np.sqrt((directions**2).sum(1))[:,None]

    cdef int ndir = directions.shape[0]

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_dir = \
        np.dot(A, directions.T)

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_coord = A
        
    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_dir = \
        np.fabs(alphas_dir).max(0) * tol    

    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_coord = \
        np.fabs(alphas_coord).max(0) * tol 

    # choose the order of sampling (randomly)

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_dir = \
        np.random.random_integers(0, ndir-1, size=(burnin+ndraw,))

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_coord = \
        np.random.random_integers(0, nvar-1, size=(burnin+ndraw,))

    # for switching between coordinate updates and
    # other directions

    cdef int invperiod = 13
    cdef int docoord = 0
    cdef int iperiod = 0
    cdef int ibias = 0
    cdef int dobias = 0

    cdef np.ndarray[DTYPE_float_t, ndim=1] _dir = \
        np.zeros_like(state)

    cdef np.ndarray[DTYPE_float_t, ndim=1] discrete_linspace = \
        np.linspace(0, 1, discretization)

    for iter_count in range(ndraw + burnin):

        docoord = 1
        iperiod = iperiod + 1
        ibias = ibias + 1

        if iperiod == invperiod: 
            docoord = 0
            iperiod = 0
            dobias = 0

        if ibias == how_often:
            docoord = 0
            ibias = 0
            dobias = 1
        
        if docoord == 1:
            idx = random_idx_coord[iter_count]
            V = state[idx]
        else:
            if not dobias:
                idx = random_idx_dir[iter_count]
            else:
                idx = directions.shape[0]-1 # last row of directions is bias_direction
            V = 0
            for ivar in range(nvar):
                V = V + directions[idx, ivar] * state[ivar]

        lower_bound = -1e12
        upper_bound = 1e12
        for irow in range(nconstraint):
            if docoord == 1:
                alpha = alphas_coord[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val
        if lower_bound > V:
            lower_bound = V - tol
        elif upper_bound < V:
            upper_bound = V + tol

        lower_bound = lower_bound
        upper_bound = upper_bound

        if lower_bound > upper_bound:
            raise ValueError('bound violation')

        # create the ray

        if docoord == 1:
            if lower_bound > -np.inf:
                base = state - (V - lower_bound) * directions[idx] 
                if upper_bound == np.inf:
                    upper_bound = lower_bound + 8
            elif upper_bound < np.inf:
                if upper_bound == np.inf:
                    lower_bound = upper_bound - 8
            X = np.multiply.outer((upper_bound - lower_bound) * discrete_linspace, directions[idx]) + base[None, :]
            D = multivariate_T_unnorm(X, degrees_of_freedom, noncentrality)
            D /= D.sum()
        else:
            cur_val = state[idx]
            _dir *= 0
            _dir[idx] = 1.

            if lower_bound > -np.inf:
                base = state - (V - lower_bound) * _dir
                if upper_bound == np.inf:
                    upper_bound = lower_bound + 8
            elif upper_bound < np.inf:
                if upper_bound == np.inf:
                    lower_bound = upper_bound - 8
            
            X = np.multiply.outer((upper_bound - lower_bound) * discrete_linspace, _dir) + base[None, :]
            D = multivariate_T_unnorm(X, degrees_of_freedom, noncentrality)
            D /= D.sum()
            
        if docoord == 1:
            state[idx] = truncT
            truncT = truncT - V
            for irow in range(nconstraint):
                U[irow] = U[irow] + truncT * A[irow, idx]
        else:
            truncT = truncT - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + truncT * directions[idx,ivar]
                for irow in range(nconstraint):
                    U[irow] = (U[irow] + A[irow, ivar] * 
                               truncT * directions[idx,ivar])

        if iter_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
    return trunc_sample

def multivariate_T_unnorm(X, degrees_of_freedom, noncentrality):
    """
    Proportional to standard multivariate T density on R^n, up to the normalizing
    constant.

    Parameters
    ----------
    
    X : np.float((*, n))
        Points at which density is evaluated.

    degrees_of_freedom : int
        Degrees of freedom

    noncentrality : np.float(n)
        Noncentrality parameter.

    Returns
    -------

    D : np.float(*)
        Density (up to normalizing constant).

    """

    _, n = X.shape

    return (1 + ((X - noncentrality[None, :])**2).sum() / degrees_of_freedom)**((n + degrees_of_freedom) / 2.)

