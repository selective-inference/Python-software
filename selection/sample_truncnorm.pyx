import numpy as np, cython
cimport numpy as np

from scipy.special import ndtr, ndtri

"""
This module has a code to sample from a truncated normal distribution
specified by a set of affine constraints.
"""

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

@cython.boundscheck(False)
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
    cdef double cdfL, cdfU, tnorm, unif, val, alpha

    cdef double tol = 1.e-4

    cdef np.ndarray[DTYPE_float_t, ndim=1] U = np.dot(A, state) - b

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)
    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx = \
        np.random.random_integers(0, nvar-1, size=(burnin+ndraw,))
    cdef np.ndarray[DTYPE_float_t, ndim=1] alpha_max = \
        np.fabs(A).max(0) * tol

    for iter_count in range(ndraw + burnin):
        idx = random_idx[iter_count]
        V = state[idx]
        lower_bound = -1e12
        upper_bound = 1e12
        for irow in range(nconstraint):
            alpha = A[irow,idx]
            val = -U[irow] / alpha + V
            if alpha > alpha_max[idx] and (val < upper_bound):
                upper_bound = val
            elif alpha < -alpha_max[idx] and (val > lower_bound):
                lower_bound = val

        if lower_bound > V:
            state[idx] = lower_bound + tol * (upper_bound - lower_bound)
        elif upper_bound < V:
            state[idx] = upper_bound + tol * (lower_bound - upper_bound)

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
            
        state[idx] = tnorm
        tnorm  = tnorm - V

        U = np.dot(A, state) - b
        #for irow in range(nconstraint):
        #    U[irow] = U[irow] + tnorm * A[irow,idx]

        if iter_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
    return trunc_sample



@cython.boundscheck(False)
def estimate_sigma_white(np.ndarray[DTYPE_float_t, ndim=2] A, 
                         np.ndarray[DTYPE_float_t, ndim=1] b, 
                         np.ndarray[DTYPE_float_t, ndim=1] initial, 
                         DTYPE_int_t nstep=100,
                         DTYPE_int_t burnin=100,
                         DTYPE_int_t ndraw=500,
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

    nstep : int
        How many gradient steps should we take?

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    sigma : np.float(nstep)
        A sequence of estimates of sigma.

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] sigmasq_sample = \
            np.empty((ndraw,), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] sigma_estimates = \
            np.empty((nstep,), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar, istep
    cdef double lower_bound, upper_bound, V, estimate
    cdef double cdfL, cdfU, tnorm, unif, val, alpha

    cdef double observed = np.linalg.norm(state)**2
    cdef sigma = 2 * np.sqrt(observed)

    cdef double tol = 1.e-4

    cdef np.ndarray[DTYPE_float_t, ndim=1] U = np.dot(A, state) - b

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)
    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx = \
        np.random.random_integers(0, nvar-1, size=(burnin+ndraw,))
    cdef np.ndarray[DTYPE_float_t, ndim=1] alpha_max = \
        np.fabs(A).max(0) * tol

    for istep in range(nstep): 

        for iter_count in range(ndraw + burnin):
            idx = random_idx[iter_count]
            V = state[idx]
            lower_bound = -1e12
            upper_bound = 1e12
            for irow in range(nconstraint):
                alpha = A[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alpha_max[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alpha_max[idx] and (val > lower_bound):
                    lower_bound = val

            if lower_bound > V:
                state[idx] = lower_bound + tol * (upper_bound - lower_bound)
            elif upper_bound < V:
                state[idx] = upper_bound + tol * (lower_bound - upper_bound)

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

            state[idx] = tnorm
            tnorm  = tnorm - V

            U = np.dot(A, state) - b
            #for irow in range(nconstraint):
            #    U[irow] = U[irow] + tnorm * A[irow,idx]

            if iter_count >= burnin:
                for ivar in range(nvar):
                    sigmasq_sample[iter_count - burnin] = \
                        (sigmasq_sample[iter_count - burnin]
                         + state[ivar]**2)

        estimate = np.mean(sigmasq_sample)
        grad = (observed - estimate) / (sigma**3)
        step = grad / (istep + 1.)
        sigma = sigma + step
        sigma_estimates[istep] = sigma

    return sigma_estimates


