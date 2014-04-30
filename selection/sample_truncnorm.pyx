import numpy as np, cython
cimport numpy as np

from libc.math cimport sin, cos, acos, asin, sqrt, fabs
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
def sample_truncnorm_white(np.ndarray[DTYPE_float_t, ndim=2] A, 
                           np.ndarray[DTYPE_float_t, ndim=1] b, 
                           np.ndarray[DTYPE_float_t, ndim=1] initial, 
                           DTYPE_float_t sigma=1.,
                           DTYPE_int_t burnin=500,
                           DTYPE_int_t ndraw=1000,
                           ):
    """
    Sample from a truncated normal with covariance
    equal to sigma**2 I.

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

    sigma : float
        Variance parameter.

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

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
    # NOT BEING USED CURRENTLY
    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack([A, 
                   np.random.standard_normal((int(nvar/5),nvar))])

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

    cdef int invperiod = 20
    cdef int docoord = 0
    cdef int iperiod = 0

    for iter_count in range(ndraw + burnin):

        iperiod = iperiod + 1
        if iperiod == invperiod:
            docoord = 0
            iperiod = 0
        else:
            docoord = 1

        docoord = 1 # other directions
                    # is buggy
        if docoord == 1:
            idx = random_idx_coord[iter_count]
            V = state[idx]
        else:
            idx = random_idx_dir[iter_count]
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
            lower_bound = V - tol * sigma
        elif upper_bound < V:
            upper_bound = V + tol * sigma

        lower_bound = lower_bound / sigma
        upper_bound = upper_bound / sigma

        if lower_bound < 0:
            cdfL = ndtr(lower_bound)
            cdfU = ndtr(upper_bound)
            unif = usample[iter_count] * (cdfU - cdfL) + cdfL
            if unif < 0.5:
                tnorm = ndtri(unif) * sigma
            else:
                tnorm = -ndtri(1-unif) * sigma
        else:
            cdfL = ndtr(-lower_bound)
            cdfU = ndtr(-upper_bound)
            unif = usample[iter_count] * (cdfL - cdfU) + cdfU
            if unif < 0.5:
                tnorm = -ndtri(unif) * sigma
            else:
                tnorm = ndtri(1-unif) * sigma
            
        if docoord == 1:
            state[idx] = tnorm
            tnorm = tnorm - V
            for irow in range(nconstraint):
                U[irow] = U[irow] + tnorm * A[irow, idx]
        else:
            tnorm = tnorm - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tnorm * directions[ivar,idx]
            for irow in range(nconstraint):
                U[irow] = (U[irow] + A[irow, ivar] * 
                           tnorm * directions[ivar,idx])

        if iter_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
    return trunc_sample

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white_sphere(np.ndarray[DTYPE_float_t, ndim=2] A, 
                                  np.ndarray[DTYPE_float_t, ndim=1] b, 
                                  np.ndarray[DTYPE_float_t, ndim=1] initial, 
                                  DTYPE_int_t burnin=500,
                                  DTYPE_int_t ndraw=1000,
                                  ):
    """
    Sample from a truncated normal with covariance
    equal to I restricted to a sphere of 
    radius `np.linalg.norm(initial)`.

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

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef np.ndarray[DTYPE_float_t, ndim=1] state_new = initial.copy()
    cdef int idx, irow, ivar
    cdef double norm_state = np.linalg.norm(state)

    cdef double Astate = 0

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw) * 2 * PI

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack([np.identity(nvar),
                   A, 
                   np.random.standard_normal((int(nvar/5),nvar))])

    directions /= np.sqrt((directions**2).sum(1))[:,None]

    cdef int ndir = directions.shape[0]

    cdef double theta, cos_theta, sin_theta_norm, dir_state, eta_norm

    # choose the order of sampling (randomly)

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_dir = \
        np.random.random_integers(0, ndir-1, size=(burnin+ndraw,))

    cdef int sample_count = 0
    cdef int iter_count = 0

    state = np.random.standard_normal((nvar,))
    state /= np.linalg.norm(state)
    state *= norm_state

    state_new = np.random.standard_normal((nvar,))
    state_new /= np.linalg.norm(state_new)
    state_new *= norm_state

    while sample_count < (ndraw + burnin):

        idx = random_idx_dir[iter_count % (ndraw + burnin)]
        #eta = directions[idx] - (directions[idx] * state).sum() * state / (norm_state**2)
#         eta_norm = 0
#         for ivar in range(nvar):
#             eta_norm = eta_norm + eta[ivar]**2
#         eta_norm = sqrt(eta_norm)
#         for ivar in range(nvar):
#             eta[ivar] = eta[ivar] / eta_norm

        eta = directions[idx] - np.dot(directions[idx], state) * state / norm_state**2
        eta /= np.linalg.norm(eta)

        theta = np.random.sample() * 2 * PI #usample[iter_count % (ndraw + burnin)] 
        cos_theta = cos(theta)
        sin_theta_norm = sin(theta) * norm_state

        # proposed state

        state_new = np.cos(theta) * state + np.sin(theta) * eta * norm_state
        #for ivar in range(nvar):
        #    state_new[ivar] = cos_theta * state[ivar] + sin_theta_norm * eta[ivar]

        slack = (np.dot(A, state_new) - b).max()
#         slack = -1e12
#         for irow in range(nconstraint):
#             Astate = 0
#             for ivar in range(nvar):
#                 Astate = Astate + A[irow,ivar] * state_new[ivar]
#             if Astate - b[irow] > slack:
#                 slack = Astate - b[irow]

        if slack < 0: # if in the set, move to it
            for ivar in range(nvar):
                state[ivar] = state_new[ivar]
            state[:] = state_new
            if sample_count >= burnin:
                for ivar in range(nvar):
                    trunc_sample[sample_count - burnin, ivar] = state_new[ivar]
                trunc_sample[sample_count - burnin,:] = state_new.copy()
            sample_count = sample_count + 1
        iter_count = iter_count + 1

    print iter_count, ndraw

    return trunc_sample

def _find_interval(a1, a2, b):
    """
    Find the interval 

    {t: a1*cos(t) + a2*sin(t) <= b}

    under the assumption that a1 <= b (i.e. the interval is non-empty).

    The assumption is not checked.

    """
    norm_a = np.sqrt(a1**2+a2**2)
    if np.fabs(b / norm_a) < 1:
        alpha = np.arcsin(a1/norm_a)
        if a2 < 0:
            alpha = PI - alpha
        soln_1 = np.arcsin(b/norm_a) - alpha
        if soln_1 < -PI:
            soln_1 = soln_1 + 2 * PI
        if soln_1 > PI:
            soln_1 = soln_1 - 2 * PI
        soln_2 = (PI - np.arcsin(b/norm_a) - alpha) 
        if soln_2 < -PI:
            soln_2 = soln_2 + 2 *PI
        if soln_2 > PI:
            soln_2 = soln_2 - 2 * PI
        if soln_2 > soln_1:
            return soln_2, soln_1
        else:
            return soln_1, soln_2
    else:
        return -PI, PI

cdef _Cfind_interval(double a1, 
                     double a2, 
                     double b):
    """
    Find the interval 

    {t: a1*cos(t) + a2*sin(t) <= b}

    under the assumption that a1 <= b (i.e. the interval is non-empty).

    The assumption is not checked.

    """
    cdef double norm_a = sqrt(a1*a1+a2*a2)
    cdef double lower, upper

    if fabs(b / norm_a) < 1:
        alpha = asin(a1/norm_a)
        if a2 < 0:
            alpha = PI - alpha
        tstar1 = asin(b/norm_a) - alpha
        tstar2 = PI - asin(b/norm_a) - alpha 
        if tstar1 > tstar2:
            lower, upper = tstar2, tstar1
        else:
            lower, upper = tstar1, tstar2
        lower, upper = sorted([tstar1, tstar2])
        tmean = 0.5 * (upper + lower)
        if a1 * cos(tmean) + a2 * sin(tmean) - b > 0:
            if upper < 0:
                lower, upper = upper, PI
            else:
                lower, upper = -PI, lower
        return lower, upper
    else:
        return -PI, PI

