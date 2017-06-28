import warnings
import numpy as np, cython
cimport numpy as np

from libc.math cimport pow, sqrt, log, exp # sin, cos, acos, asin, sqrt, fabs
from scipy.special import ndtr, ndtri

class BoundViolation(ValueError):
    pass

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
                           np.ndarray[DTYPE_float_t, ndim=1] bias_direction, #eta
                           DTYPE_int_t how_often=1000,
                           DTYPE_float_t sigma=1.,
                           DTYPE_int_t burnin=500,
                           DTYPE_int_t ndraw=1000,
                           int use_constraint_directions=1,
                           int use_random_directions=0,
                           int ignore_bound_violations=1,
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

    bias_direction : np.float
        Which projection is of most interest?

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

    use_constraint_directions : bool (optional)
        Use the directions formed by the constraints as in
        the Gibbs scheme?

    use_random_directions : bool (optional)
        Use additional random directions in
        the Gibbs scheme?

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

    if use_constraint_directions:
        _dirs = [A] 
    else:
        _dirs = []
    if use_random_directions:
        _dirs.append(np.random.standard_normal((int(nvar/5),nvar)))
    _dirs.append(bias_direction.reshape((-1, nvar)))

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack(_dirs)
        
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
    cdef int make_no_move = 0
    cdef int restart_idx = 0

    for iter_count in range(ndraw + burnin):

        make_no_move = 0

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
            lower_bound = V - tol * sigma
        elif upper_bound < V:
            upper_bound = V + tol * sigma

        lower_bound = lower_bound / sigma
        upper_bound = upper_bound / sigma

        if lower_bound > upper_bound:
            warnings.warn('bound violation')
            if not ignore_bound_violations:
                raise BoundViolation
            else:
                make_no_move = 1
            if iter_count - burnin > 0:
                restart_idx = iter_count - burnin / 2
                for ivar in range(nvar):
                    state[ivar] = trunc_sample[restart_idx, ivar] 
            else:
                for ivar in range(nvar):
                    state[ivar] = initial[ivar]

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
                        (lower_bound - upper_bound) * upper_bound)))
            tnorm = (upper_bound + log(1 - unif) / np.fabs(upper_bound)) * sigma
        elif lower_bound > 10:

            # here Z = lower_bound + E / fabs(lower_bound) (though lower_bound is positive)
            # and D = fabs((upper_bound - lower_bound) * lower_bound)
            unif = usample[iter_count] * (1 - exp(-np.fabs(
                        (upper_bound - lower_bound) * lower_bound)))
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
            
        if docoord == 1:
            state[idx] = tnorm
            tnorm = tnorm - V
            for irow in range(nconstraint):
                U[irow] = U[irow] + tnorm * A[irow, idx]
        else:
            tnorm = tnorm - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tnorm * directions[idx,ivar]
                for irow in range(nconstraint):
                    U[irow] = (U[irow] + A[irow, ivar] * 
                               tnorm * directions[idx,ivar])

        if iter_count >= burnin and not make_no_move:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
    return trunc_sample


@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white_sphere(np.ndarray[DTYPE_float_t, ndim=2] A, 
                                  np.ndarray[DTYPE_float_t, ndim=1] b, 
                                  np.ndarray[DTYPE_float_t, ndim=1] initial, 
                                  np.ndarray[DTYPE_float_t, ndim=1] bias_direction, 
                                  DTYPE_int_t how_often=1000,
                                  DTYPE_int_t burnin=500,
                                  DTYPE_int_t ndraw=1000,
                                  int use_constraint_directions=1,
                                  int use_random_directions=0,
                                  int ignore_bound_violations=1,
                                  ):
    """
    Sample from a sphere of radius `np.linalg.norm(initial)`
    intersected with a constraint.

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

    bias_direction : np.float 
        Which projection is of most interest?

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

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
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] weight_sample = \
            np.empty((ndraw,), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double tval, dval, val, alpha
    cdef double norm_state_bound_sq = np.linalg.norm(state)**2
    cdef double norm_state_sq = norm_state_bound_sq

    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] Astate = np.dot(A, state) 

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes

    if use_constraint_directions:
        _dirs = [A] 
    else:
        _dirs = []
    if use_random_directions:
        _dirs.append(np.random.standard_normal((int(nvar/5),nvar)))
    _dirs.append(bias_direction.reshape((-1, nvar)))

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack(_dirs)

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
    cdef int in_event = 0
    cdef int numout = 0
    cdef double min_multiple = 0.

    iter_count = 0

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
                val = (-Astate[irow] + b[irow]) / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = (-Astate[irow] + b[irow]) / alpha + V
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
        # below, discriminant is the square root of 
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

        if lower_bound > upper_bound:
            if not ignore_bound_violations:
                raise BoundViolation

        # sample from the line segment

        tval = lower_bound + usample[iter_count % (ndraw + burnin)] * (upper_bound - lower_bound)
            
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

        # compute squared norm of current state

        norm_state_sq = 0
        for ivar in range(nvar):
            norm_state_sq = norm_state_sq + state[ivar]*state[ivar]

        # if it escapes somehow, pull it back by projection

        if norm_state_sq > norm_state_bound_sq:
            multiplier = np.sqrt(0.999 * norm_state_bound_sq / norm_state_sq)
            for ivar in range(nvar):
                state[ivar] = state[ivar] * multiplier
            norm_state_sq = 0.999 * norm_state_bound_sq

        # check constraints

        in_event = 1
        multiplier = sqrt(norm_state_bound_sq / norm_state_sq)
        for irow in range(nconstraint):
            if Astate[irow] * multiplier > b[irow]:
                in_event = 0

        if in_event == 1:
            # store the sample

            if sample_count >= burnin:
                for ivar in range(nvar):
                    trunc_sample[sample_count-burnin, ivar] = state[ivar] * multiplier

                # now compute the smallest multiple M of state that is in the event
                # this is done by looking at each row of the affine 
                # inequalities and finding

                # \{c \geq 0: c \cdot A[i]^T state \leq b_i \right\} \cap [0,1]
                
                # the upper bound is always one because state is in the
                # event, so we need only find the lower bound,
                # which is the smallest non-negative 
                # multiple of `state` that still is in the event
            
                min_multiple = 0.
                for irow in range(nconstraint):

                    # there are 4 cases in the signs of Astate[irow] and
                    # b[irow], only this one gives a lower bound in [0,1]

                    if Astate[irow] < 0: # and b[irow] < 0: this check is not
                                         # actually necessary as this
                                         # is the only case that matters

                        val = b[irow] / Astate[irow] 
                        if min_multiple <  val:
                            min_multiple = val

                # the weight for this sample is 1 / (1-M^n)
                # because if you integrate over the ball
                # in polar coordinates integrating the radius first,
                # you get a factor of (1 - M^n) then there is the
                # integral for the point projected to the 
                # sphere

                # $$
                # \begin{aligned}
                # \int_{B \cap K} (1 - M(p(x))^n)^{-1} f(p(x)) dx &= 
                # \int_S \int_0^1 1_{\{(u,v): v \cdot u \in K\}}(y, r) 
                # (1 - M(y))^{-n} r^{n-1} f(y) dy \\		
                # &= \int_S \int_0^1 1_{\{r \in [M(y),1]\}} 
                # (1 - M(y)^n)^{-1} r^{n-1} f(y) dy \\		
                # &= \int_S \int_0^1 1_{\{r \in [M(y),1]\}} 
                # (1 - M(y)^n)^{-1} r^{n-1} f(y) dy \\		
                # \end{aligned}
                # $$

                # where $K$ is the convex set, 
                # $dy$ is surface measure on the sphere $S$
                # and $p(x)=x/\|x\|_2$

                weight_sample[sample_count-burnin] = 1 / (1 - pow(min_multiple, nvar))

            sample_count = sample_count + 1
        else:
            numout = numout + 1

        iter_count = iter_count + 1

        if sample_count >= ndraw + burnin:
            break

        # update the bound on the radius
        # this might be done by a sampler

        # norm_state_bound_sq = sample_radius_squared(state)

    return trunc_sample, weight_sample

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white_ball(np.ndarray[DTYPE_float_t, ndim=2] A, 
                                  np.ndarray[DTYPE_float_t, ndim=1] b, 
                                  np.ndarray[DTYPE_float_t, ndim=1] initial, 
                                  np.ndarray[DTYPE_float_t, ndim=1] bias_direction, 
                                  sample_radius_squared, 
                                  DTYPE_int_t how_often=1000,
                                  DTYPE_int_t burnin=500,
                                  DTYPE_int_t ndraw=1000,
                                  ):
    """
    Sample from the uniform
    distribution on a ball of given radius
    intersected with a constraint.

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

    bias_direction : np.float 
        Which projection is of most interest?

    radius : float
        Radius of ball.

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

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
    cdef double tval, val, alpha
    cdef double norm_state_bound_sq = sample_radius_squared(state)
    cdef double norm_state_sq = norm_state_bound_sq
    cdef double discriminant

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
    cdef int sample_count = 0
    cdef int in_event = 0
    cdef double min_multiple = 0.

    iter_count = 0

    while True:

        norm_state_sq = 0.
        for ivar in range(nvar):
            norm_state_sq = norm_state_sq + state[ivar]*state[ivar]

        # sample from the ball

        docoord = 1
        iperiod = iperiod + 1
        ibias = ibias + 1

        # compute V = random_direction^T state

        if iperiod == invperiod: 
            docoord = 0
            iperiod = 0
            dobias = 0

        if ibias == how_often:
            docoord = 0
            ibias = 0
            dobias = 1
        
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
                val = (-Astate[irow] + b[irow]) / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = (-Astate[irow] + b[irow]) / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val

        if lower_bound > V:
            lower_bound = V - tol 
        elif upper_bound < V:
            upper_bound = V + tol 

        # intersect the line segment with the ball

        discriminant = sqrt(V*V-(norm_state_sq-norm_state_bound_sq))
        if np.isnan(discriminant):
            upper_bound = V
            lower_bound = V
        else:
            if upper_bound > discriminant:
                upper_bound = discriminant
            if lower_bound < - discriminant:
                lower_bound = - discriminant

        # sample from the line segment

        tval = lower_bound + usample[iter_count % (ndraw + burnin)] * (upper_bound - lower_bound)
            
        # update the state and the vector dot(A, state)

        if docoord == 1:
            state[idx] = tval
            tval = tval - V
            for irow in range(nconstraint):
                Astate[irow] = Astate[irow] + tval * A[irow, idx]
        else:
            tval = tval - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tval * directions[idx,ivar]
                for irow in range(nconstraint):
                    Astate[irow] = (Astate[irow] + A[irow, ivar] * 
                                    tval * directions[idx,ivar])

        # store the sample

        if sample_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[sample_count-burnin, ivar] = state[ivar] 

        sample_count = sample_count + 1

        iter_count = iter_count + 1

        if sample_count >= ndraw + burnin:
            break

        # update the bound on the radius
        # this might be done by a sampler

        norm_state_bound_sq = sample_radius_squared(state)

    return trunc_sample

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white_ball_normal(np.ndarray[DTYPE_float_t, ndim=2] A, 
                                      np.ndarray[DTYPE_float_t, ndim=1] b, 
                                      np.ndarray[DTYPE_float_t, ndim=1] initial, 
                                      np.ndarray[DTYPE_float_t, ndim=1] bias_direction, 
                                      DTYPE_float_t radius,
                                      DTYPE_float_t sigma,
                                      DTYPE_int_t how_often=1000,
                                      DTYPE_int_t burnin=500,
                                      DTYPE_int_t ndraw=1000,
                                      ):
    """
    Sample from the centered isotropic Normal 
    distribution with variance sigma**2 on a ball of given radius
    intersected with a constraint.

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

    bias_direction : np.float 
        Which projection is of most interest?

    radius : float
        Radius of ball.

    sigma : float
        Variance parameter for centered Normal.

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

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
    cdef double cdfL, cdfU, unif, tval, val, alpha, tnorm
    cdef double norm_state_bound = np.linalg.norm(state)**2
    cdef double norm_state_sq = norm_state_bound
    cdef double discriminant

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
    cdef int sample_count = 0
    cdef int in_event = 0

    iter_count = 0

    while True:

        norm_state_sq = 0.
        for ivar in range(nvar):
            norm_state_sq = norm_state_sq + state[ivar]*state[ivar]

        # sample from the ball

        docoord = 1
        iperiod = iperiod + 1
        ibias = ibias + 1

        # compute V = random_direction^T state

        if iperiod == invperiod: 
            docoord = 0
            iperiod = 0
            dobias = 0

        if ibias == how_often:
            docoord = 0
            ibias = 0
            dobias = 1
        
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
                val = (-Astate[irow] + b[irow]) / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = (-Astate[irow] + b[irow]) / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val

        if lower_bound > V:
            lower_bound = V - tol 
        elif upper_bound < V:
            upper_bound = V + tol 

        # intersect the line segment with the ball

        discriminant = sqrt(V*V-(norm_state_sq-radius*radius))
        if np.isnan(discriminant):
            upper_bound = V
            lower_bound = V
        else:
            if upper_bound > discriminant:
                upper_bound = discriminant
            if lower_bound < - discriminant:
                lower_bound = - discriminant

        # sample along the slice

        cdfL = ndtr(-lower_bound / sigma)
        cdfU = ndtr(-upper_bound / sigma)
        unif = usample[iter_count] * (cdfL - cdfU) + cdfU
        if unif < 0.5:
            tnorm = -ndtri(unif) * sigma
        else:
            tnorm = ndtri(1-unif) * sigma
        tval = tnorm
            
        # update the state and the vector dot(A, state)

        if docoord == 1:
            state[idx] = tval
            tval = tval - V
            for irow in range(nconstraint):
                Astate[irow] = Astate[irow] + tval * A[irow, idx]
        else:
            tval = tval - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tval * directions[idx,ivar]
                for irow in range(nconstraint):
                    Astate[irow] = (Astate[irow] + A[irow, ivar] * 
                                    tval * directions[idx,ivar])

        # store the sample

        if sample_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[sample_count-burnin, ivar] = state[ivar] 

        sample_count = sample_count + 1

        iter_count = iter_count + 1

        if sample_count >= ndraw + burnin:
            break

    return trunc_sample
