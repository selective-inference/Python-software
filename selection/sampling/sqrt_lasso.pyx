import numpy as np, cython
cimport numpy as np

from libc.math cimport pow, sqrt, log, exp # sin, cos, acos, asin, sqrt, fabs
from scipy.special import ndtr, ndtri
from scipy.stats import beta, norm as ndist

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
def sample_sqrt_lasso(np.ndarray[DTYPE_float_t, ndim=2] A, 
                      np.ndarray[DTYPE_float_t, ndim=1] LHS_offset, 
                      np.ndarray[DTYPE_float_t, ndim=1] RHS_offset, 
                      np.ndarray[DTYPE_float_t, ndim=1] initial, 
                      np.ndarray[DTYPE_float_t, ndim=1] bias_direction, 
                      DTYPE_float_t RSS_max, 
                      DTYPE_int_t df_max, 
                      DTYPE_float_t RSS_1, 
                      DTYPE_int_t df_1, 
                      DTYPE_float_t sigma,
                      DTYPE_int_t how_often=1000,
                      DTYPE_int_t burnin=500,
                      DTYPE_int_t ndraw=1000,
                      ):
    """
    Sample from null distribution in sqrt LASSO.

    Constraint is $Ax + u \leq \|Rx\|_2 b$ where `A` has shape
    `(q,n)` with `q` the number of constraints and
    `n` the number of random variables and $u$ is `offset`. We assume
    implicitly that $AR=0$ and for the projection $R$ and there is
    some projection $P \geq R$ such that $AP=A, PR=R$. 

    The distribution we sample from is $N(0, \sigma^2 P)$ subject
    to the above constraints and $\|Px\|_2$ is fixed at `RSS_max`,
    with $\text{tr}(P)$ being `df_max`.

    We only store $Qx$ where $PQ=Q$ and $QR=0$. This projection 
    is the row space of the covariance in the conditional law
    in which this function is called, i.e. for data carving
    the sqrt-LASSO.

    Parameters
    ----------

    A : np.float((q,n))
        Linear part of affine constraints.

    LHS_offset : np.float(q)
        Offset part on LHS of affine constraints.

    RHS_offset : np.float(q)
        Offset part on RHS of affine constraints (is multiplied by sqrt(RSS_1))

    initial : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    bias_direction : np.float 
        Which projection is of most interest?

    RSS_1 : np.float 
        Residual sum of squares from model fit on n1 data points.

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

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

    weight_sample : np.float((ndraw,))

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar + 1), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] weight_sample = \
            np.empty((ndraw,), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double tval, dval, val, alpha
    cdef double norm_state_bound_sq = RSS_max - RSS_1
    cdef double norm_state_sq = norm_state_bound_sq
    cdef np.ndarray[DTYPE_float_t, ndim=1] effective_offset = \
            (-LHS_offset + sqrt(RSS_1) * RHS_offset)

    # we are looking at projection of uniform on sphere
    # in dimension df_max onto nvar coordinates
    
    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] Astate = np.dot(A, state) 

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack([A, 
                   np.random.standard_normal((int(nvar/5),nvar))])
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
    cdef double discriminant, multiplier
    cdef int sample_count = 0
    cdef int numout = 0
    cdef double lower_bound_RSS, upper_bound_RSS, RSS_bound_lhs

    iter_count = 0

    # this one is for RSS_1
    beta_1_rv = beta(df_1*0.5, (df_max - nvar)*0.5)

    # this is what we use to sample
    # norm_rv = lambda x: np.exp(-(x**2).sum()/2.)

    while True:

        # sample from the ball

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
        
        # V is the current value of np.dot(direction, state)

        if docoord == 1:
            idx = random_idx_coord[iter_count  % (ndraw + burnin)]
            V = state[idx]
        else:
            if not dobias:
                idx = random_idx_dir[iter_count  % (ndraw + burnin)]
            else:
                idx = directions.shape[0]-1 # last row of directions is bias_direction
            V = 0
            for ivar in range(nvar):
                V = V + directions[idx, ivar] * state[ivar]

        # compute the slice in the chosen direction

        lower_bound = -1e12
        upper_bound = 1e12
        for irow in range(nconstraint):
            if docoord == 1:
                alpha = alphas_coord[irow,idx]
                val = (-Astate[irow] + effective_offset[irow]) / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = (-Astate[irow] + effective_offset[irow]) / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val

        if lower_bound > V:
            lower_bound = V - tol 
        elif upper_bound < V:
            upper_bound = V + tol 

        # intersect the line segment with the ball
        # 
        # below, discriminant is the sqaure root of 
        # the squared overall bound on the length
        # minus the current norm of P_{\eta}^{\perp}y
        # where eta is the current direction of movement

        discriminant = sqrt(norm_state_bound_sq - (norm_state_sq - V*V))

        if np.isnan(discriminant):
            upper_bound = V
            lower_bound = V
        else:
            if upper_bound > discriminant:
                upper_bound = discriminant
            if lower_bound < - discriminant:
                lower_bound = - discriminant

        # Poincare's limit scaling says that these coordinates
        # are approx independent normals with variance 1/df_max

        # sigma = sqrt(norm_state_bound_sq / (df_max - df_1))

        lower_bound = lower_bound / sigma
        upper_bound = upper_bound / sigma

        if upper_bound < -10: # use Exp approximation
            # the approximation is that
            # Z | lower_bound < Z < upper_bound
            # is fabs(upper_bound) * (upper_bound - Z) = E approx Exp(1)
            # so Z = upper_bound - E / fabs(upper_bound)
            # and the truncation of the exponential is
            # E < fabs(upper_bound - lower_bound) * fabs(upper_bound) = D

            # this has distribution function (1 - exp(-x)) / (1 - exp(-D))
            # so to draw from this distribution
            # we set E = - log(1 - U * (1 - exp(-D))) where U is Unif(0,1)
            # and Z (= tnorm below) is as stated

            unif = usample[iter_count] * (1 - exp(-np.fabs(
                        lower_bound - upper_bound) * upper_bound))
            tnorm = (upper_bound + log(1 - unif) / np.fabs(upper_bound)) * sigma
        elif lower_bound > 10:

            # here Z = lower_bound + E / fabs(lower_bound) (though lower_bound is positive)
            # and D = fabs((upper_bound - lower_bound) * lower_bound)
            unif = usample[iter_count] * (1 - exp(-np.fabs(
                        upper_bound - lower_bound) * lower_bound))
            tnorm = (lower_bound - log(1 - unif) / lower_bound) * sigma
        elif lower_bound < 0:
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

        tval = tnorm

        # update the state vector

        if docoord == 1:
            state[idx] = tval
            dval = tval - V
            for irow in range(nconstraint):
                Astate[irow] = Astate[irow] + dval * A[irow, idx]
        else:
            dval = tval - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + dval * directions[idx,ivar]
                for irow in range(nconstraint):
                    Astate[irow] = (Astate[irow] + A[irow, ivar] * 
                                    dval * directions[idx,ivar])

        if sample_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[sample_count-burnin, ivar] = state[ivar] 
            trunc_sample[sample_count-burnin, nvar] = RSS_1

        # compute squared norm of current state

        norm_state_sq = 0
        for ivar in range(nvar):
            norm_state_sq = norm_state_sq + state[ivar]*state[ivar]

        # weight has to now be computed
        # based on the normal approximation compared to true distribution            

        weight_sample[sample_count-burnin] = (pow(1. - norm_state_sq / norm_state_bound_sq, 0.5 * (df_max - df_1)) / 
                                              exp(-norm_state_sq / (2 * sigma**2)))

        if iter_count % 30 == 0:
            # now we sample RSS_1

            lower_bound_RSS = 0.
            upper_bound_RSS = np.inf

            for irow in range(nconstraint):
                if RHS_offset[irow] > 0:
                    RSS_bound_lhs = (Astate[irow] + LHS_offset[irow]) / RHS_offset[irow]
                    if RSS_bound_lhs > lower_bound_RSS:
                        lower_bound_RSS = RSS_bound_lhs
                elif RHS_offset[irow] < 0:
                    RSS_bound_lhs = (Astate[irow] + LHS_offset[irow]) / RHS_offset[irow]
                    if RSS_bound_lhs < upper_bound_RSS:
                        upper_bound_RSS = RSS_bound_lhs

            if lower_bound_RSS > upper_bound_RSS:
                raise ValueError('RSS inequalities not satisfied')

            lower_bound_RSS = lower_bound_RSS**2
            upper_bound_RSS = min(upper_bound_RSS**2, RSS_max - norm_state_sq)

            # with the squared length of state at norm_state_sq
            # RSS_1 is between 0 and RSS_max - norm_state_sq
            # 
            # we therefore draw from beta(df_1/2, (df_max-nvar)/2)
            # truncated to [lower_bound_RSS, upper_bound_RSS]

            lower_bound_RSS = lower_bound_RSS / (RSS_max - norm_state_sq)
            upper_bound_RSS = upper_bound_RSS / (RSS_max - norm_state_sq)

            cdfL = beta_1_rv.cdf(lower_bound_RSS)
            cdfU = beta_1_rv.cdf(upper_bound_RSS)
            unif = usample[iter_count] * (cdfU - cdfL) + cdfL
            RSS_1 = beta_1_rv.ppf(unif) * (RSS_max - norm_state_sq)

            norm_state_bound_sq = RSS_max - RSS_1

        # check to see if we've drawn enough samples

        sample_count = sample_count + 1
        iter_count = iter_count + 1
        if sample_count >= ndraw + burnin:
            break


    return trunc_sample, weight_sample

