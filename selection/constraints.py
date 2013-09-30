import numpy as np
from scipy.stats import norm as ndist
from warnings import warn

class constraint(object):

    def __init__(self, 
                 inequality, 
                 equality, 
                 independent=False,
                 covariance=None):
        """
        Create a new inequality. If independent, then
        it is assumed that the rows of inequality
        and equality are independent (for a given covariance).
        """
        if equality is not None:
            self.equality, self.equality_offset = \
                np.asarray(equality[0]), equality[1]
            if self.equality.ndim == 2:
                dim_equality = self.equality.shape[1]
            else:
                dim_equality = self.equality.shape[0]
        else:
            self.equality = self.equality_offset = None

        if inequality is not None:
            self.inequality, self.inequality_offset = \
                np.asarray(inequality[0]), inequality[1]
            if self.inequality.ndim == 2:
                dim_inequality = self.inequality.shape[1]
            else:
                dim_inequality = self.inequality.shape[0]
        else:
            self.inequality = self.inequality_offset = None

        if dim_equality != dim_inequality:
            raise ValueError('constraint dimensions do not match')
        self.dim = dim_equality
        if covariance is None:
            covariance = np.identity(self.dim)
        self.covariance = covariance

    def impose_equality(self):
        """
        Return an equivalent constraint with the equality
        constraint enforced.
        """
        if self.equality is not None:
            M1 = np.dot(self.inequality, np.dot(self.covariance, 
                                                self.equality.T))
            M2 = np.dot(self.equality, np.dot(self.covariance, 
                                              self.equality.T))
            M3 = np.dot(M1, np.linalg.pinv(M2))
            
            equality_linear = np.dot(M3, self.equality)
            equality_offset = np.dot(M3, self.equality_offset)
            
            return constraint((self.inequality - equality_linear,
                               self.inequality_offset - equality_offset),
                              (self.equality, self.equality_offset),
                              independent=True)
        else:
            return self

    def __call__(self, Y, tol=1.e-3):
        """
        Check whether Y satisfies the linear
        inequality and equality constraints.
        """
        V1 = np.dot(self.inequality, Y) + self.inequality_offset
        test1 = np.all(V1 > -tol * np.fabs(V1).max())

        V2 = np.dot(self.equality, Y) + self.equality_offset
        test2 = np.linalg.norm(V2) < tol * np.linalg.norm(self.equality)
        return test1 and test2

    def pivots(self, direction_of_interest, Y):
        return interval_constraints(self.inequality,
                                    self.inequality_offset,
                                    self.covariance,
                                    Y,
                                    direction_of_interest)

def stack(*cons):
    """
    Combine constraints into a large constaint
    by intersection.
    """
    ineq, ineq_off = [], []
    eq, eq_off = [], []
    for con in cons:
        if con.inequality is not None:
            ineq.append(con.inequality)
            ineq_off.append(con.inequality_offset)
        if con.equality is not None:
            eq.append(con.equality)
            eq_off.append(con.equality_offset)

    ineq = np.vstack(ineq)
    ineq_off = np.hstack(ineq_off)

    eq = np.vstack(eq)
    eq_off = np.hstack(eq_off)
    return constraint((ineq, ineq_off), (eq, eq_off))

def simulate_from_constraints(con, tol=1.e-3):
    if con.equality is not None:
        V = np.linalg.pinv(con.equality)
        a = -np.dot(V, con.equality_offset)
        P = np.identity(con.dim) - np.dot(con.equality.T, V.T)
    else:
        a = 0
        P = np.identity(con.dim)
    if con.inequality is not None:
        while True:
            Z = np.dot(P, np.random.standard_normal(con.dim)) + a
            W = np.dot(con.inequality, Z) + con.inequality_offset  
            if np.all(W > - tol * np.fabs(W).max()):
                break
        return Z
    else:
        return np.dot(P, np.random.standard_normal(con.dim)) + a

def interval_constraints(support_directions, 
                         support_offsets,
                         covariance,
                         observed_data, 
                         direction_of_interest,
                         tol = 1.e-4):
    """
    Given an affine cone constraint $Ax+b \geq 0$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $w$, and
    an observed Gaussian vector $X$ with some `covariance`, this
    function returns $w^TX$ as well as an interval
    bounding this value. 

    The interval constructed is such that the endpoints are 
    independent of $w^TX$, hence the $p$-value
    of `Kac-Rice <http://arxiv.org/abs/1308.3020>`_
    can be used to form an exact pivot.

    """

    # shorthand
    A, b, S, X, w = (support_directions,
                     support_offsets,
                     covariance,
                     observed_data,
                     direction_of_interest)

    U = np.dot(A, X) + b
    if not np.all(U > -tol * np.fabs(U).max()):
        warn('constraints not satisfied: %s' % `U`)

    Sw = np.dot(S, w)
    sigma = np.sqrt((w*Sw).sum())
    C = np.dot(A, Sw) / sigma**2
    V = (w*X).sum()
    RHS = (-b - np.dot(A, X) + V * C) / C
    pos_coords = C > tol * np.fabs(C).max()
    if np.any(pos_coords):
        lower_bound = RHS[pos_coords].max()
    else:
        lower_bound = -np.inf
    neg_coords = C < -tol * np.fabs(C).max()
    if np.any(neg_coords):
        upper_bound = RHS[neg_coords].min()
    else:
        upper_bound = np.inf

    pval = (ndist.cdf(upper_bound / sigma) - ndist.cdf(V / sigma)) / (ndist.cdf(upper_bound / sigma) - ndist.cdf(lower_bound / sigma))
    return lower_bound, V, upper_bound, sigma, pval
