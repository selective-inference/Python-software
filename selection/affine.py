"""
This module contains the core code needed for post selection
inference based on affine selection procedures as
described in the papers `Kac Rice`_, `Spacings`_, `covTest`_
and `post selection LASSO`_.

.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238

"""

import numpy as np
from .pvalue import truncnorm_cdf
from .truncated import truncated_gaussian
                        
from warnings import warn

WARNINGS = False

class constraints(object):

    r"""
    This class is the core object for affine selection procedures.
    It is meant to describe sets of the form $C \cap E$
    where

    .. math::

       C = \left\{z: Az\leq b \right \}

       E = \left\{z: Cz = d \right\}

    Its main purpose is to consider slices through $C \cap E$
    and the conditional distribution of a Gaussian $N(\mu,\Sigma)$
    restricted to such slices.

    """

    def __init__(self, 
                 inequality, 
                 equality, 
                 covariance=None,
                 mean=None):
        r"""
        Create a new inequality. 

        Parameters
        ----------

        inequality : (A,b)
            A pair specifying the inequality constraint 
            $\{z:Az \leq b\}$. Can be `None`.

        equality: (C,d)
            A pair specifing the equality constraint
            $\{z:Cz=d\}$. Can be `None`.

        covariance : np.float
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : np.float
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
            return """$$Z \sim N(\mu,\Sigma) | AZ \leq b$$"""
        elif self.equality is not None and self.inequality is None:
            return """$$Z \sim N(\mu,\Sigma) | CZ = d$$"""
        else:
            return """$$Z \sim N(\mu,\Sigma) | AZ \leq b, CZ = d$$"""

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
           
           AY - E(AY|CZ=d)


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
        r"""
        Check whether Y satisfies the linear
        inequality and equality constraints.
        """
        if self.inequality is not None:
            V1 = np.dot(self.inequality, Y) - self.inequality_offset
            test1 = np.all(V1 < tol * np.fabs(V1).max())
        else:
            test1 = True
        if self.equality is not None:
            V2 = np.dot(self.equality, Y) - self.equality_offset
            test2 = np.linalg.norm(V2) < tol * np.linalg.norm(self.equality)
        else:
            test2 = True
        return test1 and test2

    def bounds(self, direction_of_interest, Y):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
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

        Notes
        -----
        
        This method assumes that equality constraints
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
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
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

        This method assumes that equality constraints
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
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alpha : float
            What level of confidence?

        UMAU : bool
            Use the UMAU intervals?

        Returns
        -------

        [U,L] : selection interval

        Notes
        -----
        
        This method assumes that equality constraints
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

    cons : [`selection.constraints.constraints`_]
         A sequence of constraints.

    Returns
    -------

    intersection : `selection.constraints.constraints`_

    Notes
    -----

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
    r"""
    Use naive acceptance rule to simulate from `con`.

    Parameters
    ----------

    con : `selection.constraints.constraints`_

    Notes
    -----

    This function assumes the covariance is
    proportional to the identity.

    """
    if con.equality is not None:
        V = np.linalg.pinv(con.equality)
        a = np.dot(V, con.equality_offset)
        P = np.identity(con.dim) - np.dot(con.equality.T, V.T)
    else:
        a = 0
        P = np.identity(con.dim)
    if con.inequality is not None:
        while True:
            Z = np.dot(P, np.random.standard_normal(con.dim)) + a
            Z += con.mean
            W = np.dot(con.inequality, Z) - con.inequality_offset  
            if np.all(W < tol * np.fabs(W).max()):
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
    r"""
    Given an affine in cone constraint $\{z:Az+b \leq 0\}$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
    with `covariance` matrix $\Sigma$, this
    function returns $\eta^TZ$ as well as an interval
    bounding this value. 

    The interval constructed is such that the endpoints are 
    independent of $\eta^TZ$, hence the $p$-value
    of `Kac-Rice`_
    can be used to form an exact pivot.

.. _Kac Rice: http://arxiv.org/abs/1308.3020

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    support_offset : np.float
         Offset in constraint, $b$.

    covariance : np.float
         Covariance matrix of `observed_data`.

    observed_data : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    """

    # shorthand
    A, b, S, X, w = (support_directions,
                     support_offsets,
                     covariance,
                     observed_data,
                     direction_of_interest)

    U = np.dot(A, X) - b
    if not np.all(U  < tol * np.fabs(U).max()) and WARNINGS:
        warn('constraints not satisfied: %s' % `U`)

    Sw = np.dot(S, w)
    sigma = np.sqrt((w*Sw).sum())
    alpha = np.dot(A, Sw) / sigma**2
    V = (w*X).sum() # \eta^TZ

    # adding the zero_coords in the denominator ensures that
    # there are no divide-by-zero errors in RHS
    # these coords are never used in upper_bound or lower_bound

    zero_coords = alpha == 0
    RHS = (-U + V * alpha) / (alpha + zero_coords)
    RHS[zero_coords] = np.nan

    pos_coords = alpha > tol * np.fabs(alpha).max()
    if np.any(pos_coords):
        upper_bound = RHS[pos_coords].min()
    else:
        upper_bound = np.inf
    neg_coords = alpha < -tol * np.fabs(alpha).max()
    if np.any(neg_coords):
        lower_bound = RHS[neg_coords].max()
    else:
        lower_bound = -np.inf

    return lower_bound, V, upper_bound, sigma

def selection_interval(support_directions, 
                       support_offsets,
                       covariance,
                       observed_data, 
                       direction_of_interest,
                       tol = 1.e-4,
                       alpha = 0.05,
                       UMAU=True):
    """
    Given an affine in cone constraint $\{z:Az+b \leq 0\}$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
    with `covariance` matrix $\Sigma$, this
    function returns a confidence interval
    for $\eta^T\mu$.

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    support_offset : np.float
         Offset in constraint, $b$.

    covariance : np.float
         Covariance matrix of `observed_data`.

    observed_data : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    UMAU : bool
         Use the UMAU interval, or two-sided pivot.

    Returns
    -------

    selection_interval : (float, float)

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
    
    return _selection_interval
