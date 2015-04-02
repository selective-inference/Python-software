import numpy as np
from ..constraints.affine import constraints
from .pvalue import chi_pvalue

def tangent_space(operator, y):
    """
    Return the unit vector $\eta(y)=Ay/\|Ay\|_2$ 
    where $A$ is `operator`. It also forms a basis for the 
    tangent space of the unit sphere at $\eta(y)$.

    Parameters
    ----------

    operator : np.float((p,q))

    y : np.float((q,)

    Returns
    -------

    eta : np.float(p)
        Unit vector that achieves the norm of $Ay$ with $A$ as `operator`. 

    tangent_space : np.float((p-1,p))

        An array whose rows form a basis for the tangent space
        of the sphere at `eta`.

    Notes
    -----

    Implicitly assumes $A$ is of rank p and any (p-1) rows of the identity
    can be used to find a basis of the tangent space after projection
    off of $Ay$.

    """
    A = operator # shorthand
    soln = np.dot(A, y)
    norm_soln = np.linalg.norm(soln)
    eta = np.dot(A.T, soln) / norm_soln

    p, q = A.shape
    if p > 1:
        tangent_vectors = np.identity(p)[:-1]
        for tv in tangent_vectors:
            tv[:] = tv - np.dot(tv, soln) * soln / norm_soln**2
        return eta, np.dot(tangent_vectors, A)
    else:
        return eta, None
    
def quadratic_bounds(y, operator, affine_constraints):
    r"""
    Given a set specified by an affine constraint

    .. math::

        C = \{z \in \mathbb{R}^q: Az \leq b \}

    and $A_{p \times q}$ given by `operator`, this function
    determines the slice

    .. math::

       \{t: A(y + t \eta) \leq b\}

    where

    .. math::

       \eta = \frac{A^TAy}{\|A^TAy\|_2}

    This is used to create a truncated $\chi$ test,
    as described for the group LASSO in `Kac Rice`_ and
    implemented in the function `quadratic_test`.

    Parameters
    ----------

    y : np.float((q,))

    operator : np.float((p,q))

    affine_constraints : `selection.constraints.constraints`_

    Returns
    -------

    lower_bound : float

    observed : float

    upper_bound : float

    sd  : float

    Notes
    -----

    The test is based on the fact that, conditional
    on $\eta$ and the constraints, $Ay$ is a
    truncated $\chi$ random variable.

    """
    con = affine_constraints # shorthand
    p, q = operator.shape

    eta, TA = tangent_space(operator, y)
    if TA is not None:
        newcon = constraints(con.linear_part,
                             con.offset,
                             covariance=con.covariance)
        newcon = newcon.conditional(TA, np.zeros(TA.shape[0]))
        P = np.identity(q) - np.dot(np.linalg.pinv(TA), TA)
        eta = np.dot(P, eta)
    else:
        newcon = con

    return newcon.bounds(eta, y)

def quadratic_test(y, operator, affine_constraints):
    r"""
    Test the null hypothesis $$H_0:A_{p \times q}\mu_{q \times 1} = 0$$ based on
    $y \sim N(\mu,\Sigma)$ with $\Sigma$ given by `affine_constraints.covariance`
    where `affine_constraints` represents the set

    .. math::

        C = \{z \in \mathbb{R}^q: Az \leq b \}

    Parameters
    ----------

    y : np.float((q,))

    operator : np.float((p,q))

    affine_constraints : `selection.constraints.constraints`_

    Returns
    -------

    lower_bound : float

    """
    p, q = operator.shape

    lower_bound, observed, upper_bound, sigma = quadratic_bounds(y, operator, 
                                                                 affine_constraints)
    lower_bound = max(0, lower_bound)
    
    pval = chi_pvalue(observed, lower_bound, upper_bound, sigma, p, 
                      method='MC', nsim=10000)
    return np.clip(pval, 0, 1)

