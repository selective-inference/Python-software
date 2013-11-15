"""
do marginal screening

do discrete local max

"""

def interval_constraint_max(Z, S, offset, tol=1.e-3, 
                            lower_bound=None,
                            upper_bound=None):
    '''

    Compute the maximum of Z and return 
    an interval within which it is constrained to lie. 
    The interval is constructed so that, if Z has
    covariance matrix S, then the upper and lower
    end points are independent of Z[j]
    conditional on j = j_star where Z[j_star] = np.max(Z).

    This is used for a p-value under the assumption 
    $y \sim N(\mu, \sigma^2 I)$.

    Parameters
    ----------

    Z : np.array((p,))
        Response vector.

    S : np.array((p,p))
        Covariance matrix

    offset : np.array((p,))
        Hypothesized offset of response vector.

    tol : float
        A check used to determine which values achieve the max.

    lower_bound : np.array((p,)) (optional)
        A vector of lower bound that X^Ty is constrained to lie above.

    upper_bound : np.array((p,)) (optional)
        A vector of upper bounds that X^Ty is constrained to lie below.

    Returns
    -------

    L : np.float
        Maximum of Z.

    Vplus : np.float
        A lower bound for L.

    Vminus : np.float
        An upper bound for L.

    var_star : np.float
        Variance of Z evaluated at argmax.

    offset_star : 
        Offset vector evaluated at argmax.

    '''

    Z += offset
    j_star = np.argmax(Z)
    offset_star = offset[j_star] 
    Z_star = Z[j_star]
    
    L = Z[j_star]
    var_star = S[j_star, j_star]
    C_X = S[j_star] / var_star
    
    Mplus = {}
    Mminus = {}
    keep = np.ones(Z.shape[0], np.bool)
    keep[j_star] = 0
    
    den = 1 - C_X
    num = Z - C_X * L
    Mplus = (num / den * (den > 0))[keep]
    Vplus = np.max(Mplus)

    if np.any(den < 0):
        Mminus = (num * keep / (den + (1 - keep)))[den < 0]
        Vminus = min(Mminus)
    else:
        Vminus = np.inf

    # enforce the interval constraint

    if np.any(Z > upper_bound) or np.any(Z < lower_bound):
        raise ValueError('bounds are not satisfied') 

    if DEBUG:
        print 'before:', Vplus, Vminus

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

        Vminus = min(Vminus, u_star)

    if DEBUG:
        print 'upper:', Vplus, Vminus
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

        Vplus = max(Vplus, l_star)

    if DEBUG:
        print 'after:', Vplus, Vminus

    return L, Vplus, Vminus, var_star, offset_star
 
