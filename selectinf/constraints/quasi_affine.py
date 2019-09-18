"""
This module contains the core code needed for post selection
inference based on affine selection procedures as
described in the papers `Kac Rice`_, `Spacings`_, `covTest`_
and `post selection LASSO`_.

.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238
.. _sample carving: http://arxiv.org/abs/1410.2597

"""

from warnings import warn
from copy import copy

import numpy as np

from ..truncated.T import truncated_T
from ..distributions.discrete_family import discrete_family
from mpmath import mp
import pyinter

WARNINGS = False

class constraints(object):

    r"""
    This class is the core object for quasiaffine selection procedures.
    It is meant to describe sets of the form $C$
    where

    .. math::

       C = \left\{z: Az + u \leq \|Pz\|_2b \right \}

    where $u$ is `LHS_offset`, $b$ is `RHS_offset`, $P$
    is a projection assumed to satisfy $AP=0$,
    and $A$ is `linear_part`, some fixed matrix.

    Notes
    -----

    In this parameterization, the parameter `self.mean` corresponds
    to the *reference measure* that is being truncated. It is not the
    mean of the truncated Gaussian.

    """

    def __init__(self, 
                 linear_part,
                 LHS_offset,
                 RHS_offset,
                 residual_projector,
                 covariance=None,
                 mean=None):
        r"""
        Create a new inequality. 
        
        Parameters
        ----------
        
        linear_part : np.float((q,p))
            The linear part, $A$ of the quasi-affine constraint
            $C$.

        LHS_offset: np.float(q)
            The value of $u$ in the quasi-affine constraint
            C.

        RHS_offset: np.float(q)
            The value of $b$ in the quasi-affine constraint
            C.

        residual_projector: np.float((p,p))
            The matrix $P$ above.
            C. If `covariance` is not identity, then $\|Pz\|_2$
            should be interpreted as a Mahalanobis distance.

        covariance : np.float((p,p))
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : np.float(p)
            Mean vector of Gaussian distribution to be 
            truncated. Defaults to `np.zeros(self.dim)`.

        """

        (self.linear_part, 
         self.LHS_offset, 
         self.RHS_offset,
         self.residual_projector) = (np.asarray(linear_part), 
                                     np.asarray(LHS_offset),
                                     np.asarray(RHS_offset),
                                     residual_projector)
        
        if self.linear_part.ndim == 2:
            self.dim = self.linear_part.shape[1]
        else:
            self.dim = self.linear_part.shape[0]

        if covariance is None:
            covariance = np.identity(self.dim)
        else:
            raise NotImplementedError('need to take into account nonidentity covariance for residual projector')

        self.covariance = covariance

        self.RSS_df = np.diag(self.residual_projector).sum()

        if mean is None:
            mean = np.zeros(self.dim)
        self.mean = mean

    def _repr_latex_(self):
        return """$$Z \sim N(\mu,\Sigma) | AZ + u \leq \|PZ\|_2 b$$"""

    def __copy__(self):
        r"""
        A copy of the constraints.

        Also copies _sqrt_cov, _sqrt_inv if attributes are present.
        """
        con = constraints(self.linear_part.copy(),
                          self.LHS.offset.copy(),
                          self.RHS.offset.copy(),
                          self.residual_projector.copy(),
                          mean=copy(self.mean),
                          covariance=copy(self.covariance))
        if hasattr(self, "_sqrt_cov"):
            con._sqrt_cov = self._sqrt_cov.copy()
            con._sqrt_inv = self._sqrt_inv.copy()
                          
        return con

    def _value(self, Y):
        sqrt_RSS = np.linalg.norm(np.dot(self.residual_projector, Y))
        V1 = np.dot(self.linear_part, Y) + self.LHS_offset - self.RHS_offset * sqrt_RSS
        return V1

    def __call__(self, Y, tol=1.e-3):
        sqrt_RSS = np.linalg.norm(np.dot(self.residual_projector, Y))
        V1 = np.dot(self.linear_part, Y) + self.LHS_offset - self.RHS_offset * sqrt_RSS
        return np.all(V1 < tol * np.fabs(V1).max())

    def conditional(self, linear_part, value):
        """
        Return an equivalent constraint 
        after having conditioned on a linear equality.
        
        Let the inequality constraints be specified by
        `(A,b)` and the equality constraints be specified
        by `(C,d)`. We form equivalent inequality constraints by 
        considering the residual

        Parameters
        ----------

        linear_part : np.float((k,q))
             Linear part of equality constraint, `C` above.

        value : np.float(k)
             Value of equality constraint, `b` above.

        .. math::
           
           AZ - E(AZ|CZ=d)

        Returns
        -------

        conditional_con : `constraints`
             Quasi-affine constraints having applied equality constraint.

        """

        raise NotImplementedError('class is incomplete; calculation should not assume that PZ is independent of CZ')

#         A, S = (self.linear_part, 
#                 self.covariance)
#         C, d = linear_part, value

#         M1 = np.dot(S, C.T)
#         M2 = np.dot(C, M1)
#         if M2.shape:
#             M2i = np.linalg.pinv(M2)
#             delta_cov = np.dot(M1, np.dot(M2i, M1.T))
#             delta_mean = \
#             np.dot(M1,
#                    np.dot(M2i,
#                           np.dot(C,
#                                  self.mean) - d))
#         else:
#             M2i = 1. / M2
#             delta_cov = np.multiply.outer(M1, M1) / M2i
#             delta_mean = M1 * d  / M2i

#         return constraints(self.linear_part,
#                            self.LHS_offset, 
#                            self.RHS_offset,
#                            self.residual_projector,
#                            covariance=self.covariance - delta_cov,
#                            mean=self.mean - delta_mean)

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
            A realization of $N(\mu,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        Returns
        -------

        intervals : []
            Set of truncation intervals for the $T$ statistic.

        Tobs : np.float
            The observed $T$ statistic.
       
        """

        raise NotImplementedError('class is incomplete')

        intervals, Tobs = constraints_unknown_sigma( \
            self.linear_part,
            self.RHS_offset * np.sqrt(self.RSS_df),
            self.LHS_offset, 
            Y,
            direction_of_interest,
            self.residual_projector)
        return intervals, Tobs

    def pivot(self, direction_of_interest, Y,
              alternative='greater'):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Note
        ----

        Conditions on some direction vector!

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
        by the truncated T distribution for the
        given direction $\eta$. If the alternative is 'greater'
        then we return $1-F$; if it is 'less' we return $F$
        and if it is 'twosided' we return $2 \min(F,1-F)$.
        
        """

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        intervals, Tobs = self.bounds(direction_of_interest, Y)
        truncT = truncated_T(np.array([(interval.lower_value,
                                        interval.upper_value) for interval in intervals]), self.RSS_df)
        P = float(truncT.sf(Tobs))
        if (truncT.intervals.shape == ((1,2)) and np.all(truncT.intervals == [[-np.inf, np.inf]])):
            raise ValueError('should be truncated')

        if alternative == 'greater':
            return P
        elif alternative == 'less':
            return 1 - P
        else:
            return 2 * min(P, 1-P)

    def whiten(self):
        """
        Parameters
        ----------

        Return a whitened version of constraints in a different
        basis, and a change of basis matrix.

        If `self.covariance` is rank deficient, the change-of
        basis matrix will not be square.

        """

        raise NotImplementedError('class is only defined for multiple of identity covariance')

#         if not hasattr(self, "_sqrt_cov"):
#             rank = np.linalg.matrix_rank(self.covariance)

#             D, U = np.linalg.eigh(self.covariance)
#             D = np.sqrt(D[-rank:])
#             U = U[:,-rank:]
        
#             self._sqrt_cov = U * D[None,:]
#             self._sqrt_inv = (U / D[None,:]).T

#         sqrt_cov = self._sqrt_cov
#         sqrt_inv = self._sqrt_inv

#         # original matrix is np.dot(U, U.T)

#         # NEEDS FIX residual projector should also be whitened!!

#         new_A = np.dot(self.linear_part, sqrt_cov)
#         new_con = constraints(new_A, 
#                               self.LHS_offset,
#                               self.RHS_offset,
#                               self.residual_projector)

#         mu = self.mean.copy()
#         inverse_map = lambda Z: np.dot(sqrt_cov, Z) + mu[:,None]
#         forward_map = lambda W: np.dot(sqrt_inv, W - mu)

#         return inverse_map, forward_map, new_con

class orthogonal(constraints):

    r"""
    This class is the core object for quasiaffine selection procedures.
    It is meant to describe sets of the form $C$
    where

    .. math::

       C = \left\{z: Az + u \leq \|Pz\|_2b \right \}

    where $u$ is `LHS_offset`, $b$ is `RHS_offset`, $P$
    is a projection assumed to satisfy $AP=0$,
    and $A$ is `linear_part`, some fixed matrix.

    The condition $AP=0$ is why this class is called `orthogonal`.

    Notes
    -----

    In this parameterization, the parameter `self.mean` corresponds
    to the *reference measure* that is being truncated. It is not the
    mean of the truncated Gaussian.

    """

    def __init__(self, 
                 linear_part,
                 LHS_offset,
                 RHS_offset,
                 RSS,
                 RSS_df,
                 covariance=None,
                 mean=None):
        r"""
        Create a new inequality. 
        
        Parameters
        ----------
        
        linear_part : np.float((q,p))
            The linear part, $A$ of the quasi-affine constraint
            $C$.

        LHS_offset: np.float(q)
            The value of $u$ in the quasi-affine constraint
            C.

        RHS_offset: np.float(q)
            The value of $b$ in the quasi-affine constraint
            C.

        RSS : float
            The value of $\|Pz\|_2$ above.
            If `covariance` is not identity, then $\|Pz\|_2$
            should be interpreted as a Mahalanobis distance
            relative to `self.covariance`.

        RSS_df : int
            Degrees of freedom in $\|Pz\|_2$,
            when `covariance` is a multiple of identity, then this
            should be trace(P).

        covariance : np.float((p,p))
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : np.float(p)
            Mean vector of Gaussian distribution to be 
            truncated. Defaults to `np.zeros(self.dim)`.

        """

        (self.linear_part, 
         self.LHS_offset, 
         self.RHS_offset,
         self.RSS,
         self.RSS_df) = (np.asarray(linear_part), 
                         np.asarray(LHS_offset),
                         np.asarray(RHS_offset),
                         RSS,
                         RSS_df)
        
        if self.linear_part.ndim == 2:
            self.dim = self.linear_part.shape[1]
        else:
            self.dim = self.linear_part.shape[0]

        if covariance is None:
            covariance = np.identity(self.dim)
        self.covariance = covariance

        if mean is None:
            mean = np.zeros(self.dim)
        self.mean = mean

    def _repr_latex_(self):
        return """$$Z \sim N(\mu,\Sigma) | AZ + u \leq \|PZ\|_2 b$$"""

    def __copy__(self):
        r"""
        A copy of the constraints.

        Also copies _sqrt_cov, _sqrt_inv if attributes are present.
        """
        con = orthogonal(self.linear_part.copy(),
                         self.LHS.offset.copy(),
                         self.RHS.offset.copy(),
                         copy(self.RSS),
                         copy(self.RSS_df),
                         mean=copy(self.mean),
                         covariance=copy(self.covariance))
        if hasattr(self, "_sqrt_cov"):
            con._sqrt_cov = self._sqrt_cov.copy()
            con._sqrt_inv = self._sqrt_inv.copy()
                          
        return con

    def __call__(self, Y, tol=1.e-3):
        V1 = np.dot(self.linear_part, Y) + self.LHS_offset - self.RHS_offset * np.sqrt(self.RSS)
        return np.all(V1 < tol * np.fabs(V1).max())

    def conditional(self, linear_part, value):
        """
        Return an equivalent constraint 
        after having conditioned on a linear equality.
        
        Let the inequality constraints be specified by
        `(A,b)` and the equality constraints be specified
        by `(C,d)`. We form equivalent inequality constraints by 
        considering the residual

        Parameters
        ----------

        linear_part : np.float((k,q))
             Linear part of equality constraint, `C` above.

        value : np.float(k)
             Value of equality constraint, `b` above.

        .. math::
           
           AZ - E(AZ|CZ=d)

        Returns
        -------

        conditional_con : `orthogonal`
             Quasi-affine constraints having applied equality constraint.

        Notes
        -----

        The calculations here assume that $CZ$ is independent of $PZ$.

        """

        A, S = (self.linear_part, 
                self.covariance)
        C, d = linear_part, value

        M1 = np.dot(S, C.T)
        M2 = np.dot(C, M1)
        if M2.shape:
            M2i = np.linalg.pinv(M2)
            delta_cov = np.dot(M1, np.dot(M2i, M1.T))
            delta_mean = \
            np.dot(M1,
                   np.dot(M2i,
                          np.dot(C,
                                 self.mean) - d))
        else:
            M2i = 1. / M2
            delta_cov = np.multiply.outer(M1, M1) / M2i
            delta_mean = M1 * d  / M2i

        return orthogonal(self.linear_part,
                          self.LHS_offset, 
                          self.RHS_offset,
                          self.RSS,
                          self.RSS_df,
                          covariance=self.covariance - delta_cov,
                          mean=self.mean - delta_mean)

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
            A realization of $N(\mu,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        Returns
        -------

        intervals : []
            Set of truncation intervals for the $T$ statistic.

        Tobs : np.float
            The observed $T$ statistic.
       
        """

        intervals, Tobs = constraints_unknown_sigma( \
            self.linear_part,
            self.RHS_offset * np.sqrt(self.RSS_df),
            self.LHS_offset, 
            Y,
            direction_of_interest,
            self.RSS,
            self.RSS_df)
        return intervals, Tobs

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
        by the truncated T distribution for the
        given direction $\eta$. If the alternative is 'greater'
        then we return $1-F$; if it is 'less' we return $F$
        and if it is 'twosided' we return $2 \min(F,1-F)$.
        
        """

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        intervals, Tobs = self.bounds(direction_of_interest, Y)
        truncT = truncated_T(np.array([(interval.lower_value,
                                        interval.upper_value) for interval in intervals]), self.RSS_df)
        P = float(truncT.sf(Tobs))
        if (truncT.intervals.shape == ((1,2)) and np.all(truncT.intervals == [[-np.inf, np.inf]])):
            raise ValueError('should be truncated')

        if alternative == 'greater':
            return P
        elif alternative == 'less':
            return 1 - P
        else:
            return 2 * min(P, 1-P)

    def whiten(self):
        """
        Parameters
        ----------

        Return a whitened version of constraints in a different
        basis, and a change of basis matrix.

        If `self.covariance` is rank deficient, the change-of
        basis matrix will not be square.

        """

        if not hasattr(self, "_sqrt_cov"):
            rank = np.linalg.matrix_rank(self.covariance)

            D, U = np.linalg.eigh(self.covariance)
            D = np.sqrt(D[-rank:])
            U = U[:,-rank:]
        
            self._sqrt_cov = U * D[None,:]
            self._sqrt_inv = (U / D[None,:]).T

        sqrt_cov = self._sqrt_cov
        sqrt_inv = self._sqrt_inv

        # original matrix is np.dot(U, U.T)

        new_linear = np.dot(self.linear_part, sqrt_cov)
        new_con = orthogonal(new_linear, 
                             self.LHS_offset + np.dot(self.linear_part, self.mean),
                             self.RHS_offset,
                             self.RSS,
                             self.RSS_df)

        mu = self.mean.copy()
        inverse_map = lambda Z: np.dot(sqrt_cov, Z) + mu[:,None]
        forward_map = lambda W: np.dot(sqrt_inv, W - mu)

        return inverse_map, forward_map, new_con

def stack(*cons):
    """
    Combine constraints into a large constaint
    by intersection. 

    Parameters
    ----------

    cons : [`selection.affine.constraints`_]
         A sequence of constraints.

    Returns
    -------

    intersection : `selection.quasi_affine.constraints`_

    Notes
    -----

    Resulting constraint will have mean 0 and covariance $I$.
    If each is of type `constraints`, then quietly assumes that all residual projectors
    are the same, so it uses the first residual projector
    in the stack. If they are of type `orthogonal` then quietly
    assumes that all RSS and RSS_df are the same.

    If they are of mixed type, raises an exception.
    """
    ineq, ineq_LHS_off, ineq_RHS_off = [], [], []

    if np.all([isinstance(con, constraints) for con in cons]):
        for con in cons:
            ineq.append(con.linear_part)
            ineq_LHS_off.append(con.LHS_offset)
            ineq_RHS_off.append(con.RHS_offset)
        intersection = constraints(np.vstack(ineq), 
                                   np.hstack(ineq_LHS_off),
                                   np.hstack(ineq_RHS_off),
                                   cons[0].residual_projector
                                   )
    elif np.all([isinstance(con, orthogonal) for con in cons]):
        for con in cons:
            ineq.append(con.linear_part)
            ineq_LHS_off.append(con.LHS_offset)
            ineq_RHS_off.append(con.RHS_offset)
        intersection = constraints(np.vstack(ineq), 
                                   np.hstack(ineq_LHS_off),
                                   np.hstack(ineq_RHS_off),
                                   cons[0].RSS,
                                   cons[0].RSS_df
                                   )
    else:
        raise ValueError('all constraints must of same type')
    return intersection

def sample_from_constraints(con, 
                            Y,
                            direction_of_interest=None,
                            how_often=-1,
                            ndraw=1000,
                            burnin=1000,
                            white=False,
                            use_constraint_directions=True):
    r"""
    Use Gibbs sampler to simulate from `con`.

    Parameters
    ----------

    con : `selection.affine.constraints`_

    Y : np.float
        Point satisfying the constraint.

    direction_of_interest : np.float (optional)
        Which projection is of most interest?

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

    ndraw : int (optional)
        Defaults to 1000.

    burnin : int (optional)
        Defaults to 1000.

    white : bool (optional)
        Is con.covariance equal to identity?

    use_constraint_directions : bool (optional)
        Use the directions formed by the constraints as in
        the Gibbs scheme?

    Returns
    -------

    Z : np.float((ndraw, n))
        Sample from the sphere intersect the constraints.
        
    """

    raise NotImplementedError("first get the sphere sampler working.")

    # this will be different than data carving sqrtlasso

    if direction_of_interest is None:
        direction_of_interest = np.random.standard_normal(Y.shape)
    if how_often < 0:
        how_often = ndraw + burnin

    if not white:
        inverse_map, forward_map, white_con = con.whiten()
        white_Y = forward_map(Y)
        white_direction_of_interest = forward_map(np.dot(con.covariance, direction_of_interest))
    else:
        white_con = con
        inverse_map = lambda V: V

    white_samples = sample_truncnorm_white(white_con.linear_part,
                                           white_con.offset,
                                           white_Y, 
                                           white_direction_of_interest,
                                           how_often=how_often,
                                           ndraw=ndraw, 
                                           burnin=burnin,
                                           sigma=1.,
                                           use_A=use_constraint_directions)
    Z = inverse_map(white_samples.T).T
    return Z

def sample_from_sphere(con, 
                       Y,
                       direction_of_interest=None,
                       how_often=-1,
                       ndraw=1000,
                       burnin=1000,
                       white=False):
    r"""
    Use Gibbs sampler to simulate from `con` 
    intersected with (whitened) sphere of radius `np.linalg.norm(Y)`.
    When `con.covariance` is not $I$, it samples from the
    ellipse of constant Mahalanobis distance from `con.mean`.

    Parameters
    ----------

    con : `selection.affine.constraints`_

    Y : np.float
        Point satisfying the constraint.

    direction_of_interest : np.float (optional)
        Which projection is of most interest?

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

    ndraw : int (optional)
        Defaults to 1000.

    burnin : int (optional)
        Defaults to 1000.

    white : bool (optional)
        Is con.covariance equal to identity?

    Returns
    -------

    Z : np.float((ndraw, n))
        Sample from the sphere intersect the constraints.
        
    weights : np.float(ndraw)
        Importance weights for the sample.

    """

    # this is data carving sqrt_lasso

    if direction_of_interest is None:
        direction_of_interest = np.random.standard_normal(Y.shape)
    if how_often < 0:
        how_often = ndraw + burnin

    if not white:
        inverse_map, forward_map, white_con = con.whiten()
        white_Y = forward_map(Y)
        white_direction_of_interest = forward_map(direction_of_interest)
    else:
        white_con = con
        inverse_map = lambda V: V

    RSS = np.linalg.norm(np.dot(white_con.residual_projector, white_Y))

    white_samples, weights = sample_quasi_white_sphere(white_con.linear_part,
                                                       white_con.RHS_offset,
                                                       white_con.LHS_offset,
                                                       white_Y, 
                                                       white_direction_of_interest,
                                                       np.linalg.norm(white_Y)**2,
                                                       white_con.dim,
                                                       RSS,
                                                       white_con.RSS_df,
                                                       how_often=how_often,
                                                       ndraw=ndraw, 
                                                       burnin=burnin)

    Z = inverse_map(white_samples.T).T
    return Z, weights

def gibbs_test(quasi_affine_con, 
               Y, 
               direction_of_interest,
               how_often=-1,
               ndraw=5000,
               burnin=2000,
               white=False,
               alternative='twosided',
               UMPU=True,
               sigma_known=False,
               alpha=0.05,
               use_constraint_directions=False):
    """
    A Monte Carlo significance test for
    a given function of `con.mean`.

    Parameters
    ----------

    quasi_affine_con : `orthogonal`

    Y : np.float
        Point satisfying the constraint.

    direction_of_interest: np.float
        Which linear function of `con.mean` is of interest?
        (a.k.a. $\eta$ in many of related papers)

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

    ndraw : int (optional)
        Defaults to 1000.

    burnin : int (optional)
        Defaults to 1000.

    white : bool (optional)
        Is con.covariance equal to identity?

    alternative : str
        One of ['greater', 'less', 'twosided']

    UMPU : bool
        Perform the UMPU test?

    sigma_known : bool
        Is $\sigma$ assumed known?

    alpha : 
        Level for UMPU test.

    use_constraint_directions : bool (optional)
        Use the directions formed by the constraints as in
        the Gibbs scheme?

    Returns
    -------

    pvalue : float
        P-value (using importance weights) for specified hypothesis test.

    Z : np.float((ndraw, n))
        Sample from the sphere intersect the constraints.
        
    weights : np.float(ndraw)
        Importance weights for the sample.
    """

    eta = direction_of_interest # shorthand

    if alternative not in ['greater', 'less', 'twosided']:
        raise ValueError("expecting alternative to be in ['greater', 'less', 'twosided']")

    if not sigma_known:
        Z, W = sample_from_sphere(affine_con,
                                  Y,
                                  eta,
                                  how_often=how_often,
                                  ndraw=ndraw,
                                  burnin=burnin,
                                  white=white)
    else:
        Z = sample_from_constraints(affine_con,
                                    Y,
                                    eta,
                                    how_often=how_often,
                                    ndraw=ndraw,
                                    burnin=burnin,
                                    white=white,
                                    use_constraint_directions=\
                                        use_constraint_directions)
        W = np.ones(Z.shape[0], np.float)

    null_statistics = np.dot(Z, eta)
    observed = (eta*Y).sum()
    if alternative == 'greater':
        pvalue = (W*(null_statistics >= observed)).sum() / W.sum()
    elif alternative == 'less':
        pvalue = (W*(null_statistics <= observed)).sum() / W.sum()
    elif not UMPU:
        pvalue = (W*(null_statistics <= observed)).sum() / W.sum()
        pvalue = 2 * min(pvalue, 1 - pvalue)
    else:
        dfam = discrete_family(null_statistics, W)
        decision = dfam.two_sided_test(0, observed, alpha=alpha)
        return decision, Z, W
    return pvalue, Z, W

def constraints_unknown_sigma( \
    support_directions, 
    RHS_offsets,
    LHS_offsets,
    observed_data, 
    direction_of_interest,
    RSS,
    RSS_df,
    value_under_null=0.,
    tol = 1.e-4,
    DEBUG=False):
    r"""
    Given a quasi-affine constraint $\{z:Az+u \leq \hat{\sigma}b\}$ 
    (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    an `observed_data` is Gaussian vector $Z \sim N(\mu,\sigma^2 I)$ 
    with $\sigma$ unknown, this
    function returns $\eta^TZ$ as well as a set
    bounding this value. The value of $\hat{\sigma}$ is taken to be
    sqrt(RSS/RSS_df)

    The interval constructed is such that the endpoints are 
    independent of $\eta^TZ$, hence the 
    selective $T$ distribution of
    of `sample carving`_
    can be used to form an exact pivot.

    To construct the interval, we are in effect conditioning
    on all randomness perpendicular to the direction of interest,
    i.e. $P_{\eta}^{\perp}X$ where $X$ is the Gaussian data vector.

    Notes
    -----

    Covariance is assumed to be an unknown multiple of the identity.

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    RHS : np.float
         Offset in constraint, $b$.

    LHS_offsets : np.float
         Offset in LHS of constraint, $u$.

    observed_data : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    RSS : float
        Residual sum of squares.

    RSS_df : int
        Degrees of freedom of RSS.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    Returns
    -------

    lower_bound : float

    observed : float

    upper_bound : float

    sigma : float

    """

    # shorthand
    A, b, L, X, w, theta = (support_directions,
                            RHS_offsets,
                            LHS_offsets,
                            observed_data,
                            direction_of_interest,
                            value_under_null)

    # make direction of interest a unit vector

    normw = np.linalg.norm(w)
    w = w / normw
    theta = theta / normw

    sigma_hat = np.sqrt(RSS / RSS_df)

    # compute the sufficient statistics

    U = (w*X).sum() - theta
    V = X - (X*w).sum() * w
    W = sigma_hat**2 * RSS_df + U**2
    Tobs = U / np.sqrt((W - U**2) / RSS_df)
    sqrtW = np.sqrt(W)
    alpha = np.dot(A, w)

    gamma = theta * alpha + np.dot(A, V) + L

    Anorm = np.fabs(A).max()

    intervals = []
    intervals = []
    for _a, _b, _c in zip(alpha, b, gamma):
        _a = _a * sqrtW
        _b = _b * sqrtW
        cur_intervals = sqrt_inequality_solver(_a, _c, _b, RSS_df)
        intervals.append(pyinter.IntervalSet([pyinter.closed(*i) for i in cur_intervals if i]))

    truncation_set = intervals[0]
    for interv in intervals[1:]:
        truncation_set = truncation_set.intersection(interv)
    if not truncation_set:
        raise ValueError("empty truncation intervals")
    return truncation_set, Tobs

def quadratic_inequality_solver(a, b, c, direction="less than"):
    '''
    solves a * x**2 + b * x + c \leq 0, if direction is "less than",
    solves a * x**2 + b * x + c \geq 0, if direction is "greater than",
    
    returns:
    the truancated interval, may include [-infty, + infty]
    the returned interval(s) is a list of disjoint intervals indicating the union.
    when the left endpoint of the interval is equal to the right, return empty list 
    '''
    if direction not in ["less than", "greater than"]:
        raise ValueError("direction should be in ['less than', 'greater than']")
    
    if direction == "less than":
        d = b**2 - 4*a*c
        if a > 0:
            if d <= 0:
                #raise ValueError("No valid solution")
                return [[]]
            else:
                lower = (-b - np.sqrt(d)) / (2*a)
                upper = (-b + np.sqrt(d)) / (2*a)
                return [[lower, upper]]
        elif a < 0:
            if d <= 0:
                return [[float("-inf"), float("inf")]]
            else:
                lower = (-b + np.sqrt(d)) / (2*a)
                upper = (-b - np.sqrt(d)) / (2*a)
                return [[float("-inf"), lower], [upper, float("inf")]]
        else:
            if b > 0:
                return [[float("-inf"), -c/b]]
            elif b < 0:
                return [[-c/b, float("inf")]]
            else:
                raise ValueError("Both coefficients are equal to zero")
    else:
        return quadratic_inequality_solver(-a, -b, -c, direction="less than")


def intersection(I1, I2):
    if (not I1) or (not I2) or min(I1[1], I2[1]) <= max(I1[0], I2[0]):
        return []
    else:
        return [max(I1[0], I2[0]), min(I1[1], I2[1])]

def sqrt_inequality_solver(a, b, c, n):
    '''
    find the intervals for t such that,
    a*t + b*sqrt(n + t**2) \leq c

    returns:
    should return a single interval
    '''
    if b >= 0:
        intervals = quadratic_inequality_solver(b**2 - a**2, 2*a*c, b**2 * n - c**2)
        if a > 0:
            '''
            the intervals for c - at \geq 0 is
            [-inf, c/a]
            '''
            return [intersection(I, [float("-inf"), c/a]) for I in intervals]
        elif a < 0:
            '''
            the intervals for c - at \geq 0 is
            [c/a, inf]
            '''
            return [intersection(I, [c/a, float("inf")]) for I in intervals]
        elif c >= 0:
            return intervals
        else:
            return [[]]
    else:
        '''
        the intervals we will return is {c - at \geq 0} union
        {c - at \leq 0} \cap {quadratic_inequality_solver(b**2 - a**2, 2*a*c, b**2 * n - c**2, "greater than")}
        '''
        intervals = quadratic_inequality_solver(b**2 - a**2, 2*a*c, b**2 * n - c**2, "greater than")
        if a > 0:
            return [intersection(I, [c/a, float("inf")]) for I in intervals] + [[float("-inf"), c/a]]
        elif a < 0:
            return [intersection(I, [float("-inf"), c/a]) for I in intervals] + [[c/a, float("inf")]]
        elif c >= 0:
            return [[float("-inf"), float("inf")]]
        else:
            return intervals


