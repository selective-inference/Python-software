import numpy as np
from .intervals import pivot
from .truncated import truncated_gaussian, truncnorm_cdf
from warnings import warn

WARNINGS = False

class constraints(object):

    def __init__(self, 
                 inequality, 
                 equality, 
                 covariance=None,
                 mean=None):
        """
        Create a new inequality. 

        Parameters:
        -----------

        inequality : (A,b)
            A pair specifying the inequality constraint 
            $\{z:Az+b \geq 0\}$. Can be `None`.

        equality: (C,d)
            A pair specifing the equality constraint
            $\{z:Cz+d=0\}$. Can be `None`.

        covariance : `np.float`
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : `np.float`
            Mean vector of Gaussian distribution to be 
            truncated. Defaults to `np.zeros(self.dim)`.

        """
        if equality is not None:
            self.equality, self.equality_offset = \
                np.asarray(equality[0]), equality[1]
            if self.equality.ndim == 2:
                dim_equality = self.equality.shape[1]
            else:
                dim_equality = self.equality.shape[0]
        else:
            self.equality = self.equality_offset = dim_equality = None

        if inequality is not None:
            self.inequality, self.inequality_offset = \
                np.asarray(inequality[0]), inequality[1]
            if self.inequality.ndim == 2:
                dim_inequality = self.inequality.shape[1]
            else:
                dim_inequality = self.inequality.shape[0]
        else:
            self.inequality = self.inequality_offset = dim_inequality = None

        if ((dim_equality is not None) and
            (dim_inequality is not None) and
            (dim_equality != dim_inequality)):
            raise ValueError('constraint dimensions do not match')

        if dim_equality is not None:
            self.dim = dim_equality
        else:
            self.dim = dim_inequality

        if covariance is None:
            covariance = np.identity(self.dim)
        self.covariance = covariance

        if mean is None:
            mean = np.zeros(self.dim)
        self.mean = mean

    def _repr_latex_(self):
        if self.inequality is not None and self.equality is None:
            return """$$Z \sim N(\mu,\Sigma) | AZ+b \geq 0$$"""
        elif self.equality is not None and self.inequality is None:
            return """$$Z \sim N(\mu,\Sigma) | CZ+d = 0$$"""
        else:
            return """$$Z \sim N(\mu,\Sigma) | AZ+b \geq 0, CZ+d = 0$$"""

    def impose_equality(self):
        """
        Return an equivalent constraint with a
        new inequality constraint which has equality
        constraint enforced.
        
        Let the inequality constraints be specified by
        `(A,b)` and the inequality constraints be specified
        by `(C,d)`. We form equivalent inequality constraints by 
        considering the residual

        .. math::
           
           AY - E(AY|CZ+d=0)


        """
        if self.equality is not None:
            M1 = np.dot(self.inequality, np.dot(self.covariance, 
                                                self.equality.T))
            M2 = np.dot(self.equality, np.dot(self.covariance, 
                                              self.equality.T))
            M3 = np.dot(M1, np.linalg.pinv(M2))
            
            equality_linear = np.dot(M3, self.equality)
            equality_offset = np.dot(M3, self.equality_offset)
            
            return constraints((self.inequality - equality_linear,
                               self.inequality_offset - equality_offset),
                              (self.equality, self.equality_offset),
                               covariance=self.covariance)
        else:
            return self

    def __call__(self, Y, tol=1.e-3):
        """
        Check whether Y satisfies the linear
        inequality and equality constraints.
        """
        if self.inequality is not None:
            V1 = np.dot(self.inequality, Y) + self.inequality_offset
            test1 = np.all(V1 > -tol * np.fabs(V1).max())
        else:
            test1 = True
        if self.equality is not None:
            V2 = np.dot(self.equality, Y) + self.equality_offset
            test2 = np.linalg.norm(V2) < tol * np.linalg.norm(self.equality)
        else:
            test2 = True
        return test1 and test2

    def bounds(self, direction_of_interest, Y):
        """
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$.

        Parameters
        ----------

        direction_of_interest: `np.float`
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : `np.float`
            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        Returns
        -------

        L : np.float
            Lower truncation bound.

        Z : np.float
            The observed $\eta^TY$

        U : np.float
            Upper truncation bound.

        S : np.float
            Standard deviation of $\eta^TY$.

        WARNING
        -------
        
        This implicitly assumes that equality constraints
        have been enforced and direction of interest
        is in the row space of any equality constraint matrix.
        
        """
        return interval_constraints(self.inequality,
                                    self.inequality_offset,
                                    self.covariance,
                                    Y,
                                    direction_of_interest)

    def pivot(self, direction_of_interest, Y,
              alternative='greater'):
        """
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: `np.float`
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : `np.float`
            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------

        P : np.float
            $p$-value of corresponding test.

        Notes
        -----

        All of the tests are based on the exact pivot $F$ given
        by the truncated Gaussian distribution for the
        given direction $\eta$. If the alternative is 'greater'
        then we return $1-F$; if it is 'less' we return $F$
        and if it is 'twosided' we return $2 \min(F,1-F)$.

        WARNING
        -------
        
        This implicitly assumes that equality constraints
        have been enforced and direction of interest
        is in the row space of any equality constraint matrix.
        
        """
        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")
        L, Z, U, S = self.bounds(direction_of_interest, Y)
        meanZ = (direction_of_interest * self.mean).sum()
        P = truncnorm_cdf((Z-meanZ)/S, (L-meanZ)/S, (U-meanZ)/S)
        if alternative == 'greater':
            return 1 - P
        elif alternative == 'less':
            return P
        else:
            return 2 * min(P, 1-P)

    def interval(self, direction_of_interest, Y,
                 alpha=0.05, UMAU=False):
        """
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: `np.float`
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : `np.float`
            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alpha : float
            What level of confidence?

        UMAU : bool
            Use the UMAU intervals?

        Returns
        -------

        [U,L] : selection interval

        WARNING
        -------
        
        This implicitly assumes that equality constraints
        have been enforced and direction of interest
        is in the row space of any equality constraint matrix.
        
        """

        return selection_interval( \
            self.inequality,
            self.inequality_offset,
            self.covariance,
            Y,
            direction_of_interest,
            alpha=alpha,
            UMAU=UMAU)

def stack(*cons):
    """
    Combine constraints into a large constaint
    by intersection. 

    Parameters
    ----------

    cons : [`constraints`]
         A sequence of constraints.

    Returns
    -------

    intersection : `constraints`

    WARNING
    -------

    Resulting constraint will have mean 0 and covariance $I$.

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

    if ineq and eq:
        intersection = constraints((np.vstack(ineq), 
                                    np.hstack(ineq_off)), 
                                   (np.vstack(eq), 
                                    np.hstack(eq_off)))
    elif eq:
        intersection = constraints(None, 
                                   (np.vstack(eq), 
                                    np.hstack(eq_off)))
    elif ineq:
        intersection = constraints((np.vstack(ineq), 
                                    np.hstack(ineq_off)), None)
    return intersection

def simulate_from_constraints(con, tol=1.e-3):
    """
    Use naive acceptance rule to simulate from `con`.

    WARNING
    -------

    This function implicitly assuems the covariance is
    proportional to the identity.

    """
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
            Z += con.mean
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
    Given an affine in cone constraint $Ax+b \geq 0$ (elementwise)
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
    if not np.all(U > -tol * np.fabs(U).max()) and WARNINGS:
        warn('constraints not satisfied: %s' % `U`)

    Sw = np.dot(S, w)
    sigma = np.sqrt((w*Sw).sum())
    C = np.dot(A, Sw) / sigma**2
    V = (w*X).sum()

    # adding the zero_coords in the denominator ensures that
    # there are no divide-by-zero errors in RHS
    # these coords are never used in upper_bound or lower_bound

    zero_coords = C == 0
    RHS = (-U + V * C) / (C + zero_coords)
    RHS[zero_coords] = np.nan

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

    return lower_bound, V, upper_bound, sigma

def selection_interval(support_directions, 
                       support_offsets,
                       covariance,
                       observed_data, 
                       direction_of_interest,
                       tol = 1.e-4,
                       dps = 100,
                       alpha = 0.05,
                       UMAU=True):
    """
    Given an affine in cone constraint $Ax+b \geq 0$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $w$, and
    an observed Gaussian vector $X$ with some `covariance`, this
    function returns a confidence interval
    for $w^T\mu$.

    """

    lower_bound, V, upper_bound, sigma = interval_constraints( \
        support_directions, 
        support_offsets,
        covariance,
        observed_data, 
        direction_of_interest,
        tol=tol)

    if UMAU:
        truncated = truncated_gaussian([(lower_bound, upper_bound)], sigma=sigma)
        truncated.use_R = False
        _selection_interval = truncated.UMAU_interval(V, alpha)

    else:
        _selection_interval = truncated.naive_interval(V, alpha)
        # interval(lower_bound, V, upper_bound, sigma,
#                                        upper_target=1-alpha/2,
#                                        lower_target=alpha/2,
#                                        dps=dps)
    
    return _selection_interval
