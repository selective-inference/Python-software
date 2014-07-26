import numpy as np

import intervals
intervals = reload(intervals)
from intervals import intervals

from time import time

class quad_constraints(object):

    r"""

    Object solving the problems of slices for quadratic and linear 
    inequalities

    .. math::
    
          \forall i, y^T Q_i y + a_i^t < b_i


    """
    
    def __init__(self, quad_part, lin_part=None, offset=None):
        r"""

        Create a new object for quadratic constraints

        Parameters
        ----------

        quad_part : np.float(l, p, p)
              3-dimensions array. The lines are some quadratic forms
 
        lin_part : np.float(l, p)
              The lines are the vector of the linear forms in the inquality
              Default to np.zeros(l, p)

        offset : np.float(l)
              The offsets of all inequalities
              Defaults to np.zeros(l)

        WARNING : The shapes of the three parameters must fit
        """
              
        # Check the inputs are aligned
        p, _ = quad_part[0].shape
        l = len(quad_part)
        
        if lin_part == None:
            lin_part = np.zeros((l, p))
        if offset == None:
            offset = np.zeros(l)
       
    
        if len(lin_part) != l or len(offset) != l:
            raise ValueError(
                "Not the same number of quadratics, linear and offset")
        for q in [q for q in quad_part if q.shape != (p, p)]:
            raise ValueError("The quadratics must have the same shape")
        for a in [a for a in lin_part if a.shape != (p,)]:
            raise ValueError("The linear don't fit")
                

        self.quad_part = quad_part
        self.lin_part = np.array(lin_part)
        
        self.offset = offset


    def __call__(self, y, tol=1.e-3):
        """
        Check wether y satisfies the quadratic inequality constraints

        Parameters
        ----------

        y : np.float(p, 1)
              the vector tested

        tol : float
              tolerance for the inequlity. 
              Default to 1.e-3

        Returns
        -------

        bool:
              True if y satisfies the inequalities, else, False

        
        >>> q1 = np.identity(2)
        >>> q2 = np.array([[1, 0], [0, -1]])
        >>> q3 = np.identity(2)
        >>> lin_part = [np.zeros(2), np.array([0, -4]), np.zeros(2)]
        >>> off = np.array([4, 1, 5])
        >>> cons = quad_constraints([q1, q2, q3], lin_part, off)

        >>> y1 = np.array([[1.5, 0. ]])
        >>> y2 = np.array([[0. , 1.5]])
        >>> y3 = np.array([[0., -1.5]]) 
        >>> y4 = np.array([[0. , 3. ]])
        >>> cons(y1), cons(y2), cons(y3), cons(y4)
        (False, True, False, False)
        
        """

        V1 =  np.dot(y.T, np.dot(self.quad_part, y)).reshape(-1) + \
              np.dot(self.lin_part, y).reshape(-1) - self.offset
        return np.all(V1 < tol * np.linalg.norm(V1, ord = np.inf))

    def bounds(self, eta, y):
        """
        Return the intervals of the slice in a direction eta, which respects the inequality

        Parameters
        ----------
        
        eta : np.float(p)
              The direction of the slice

        y : np.float(p)
              A point on the affine slice

        Returns
        -------
        intervals : array of couples
              Array of (a, b), which means that the set is the union
              of [a, b]

        """
        
        interv = intervals()
        I_l, I_u = -np.inf, np.inf
        L, U = -np.inf, np.inf

        for M, A, off in zip(self.quad_part, self.lin_part, self.offset):

            a = float(  np.dot(eta.T, np.dot(M, eta))  )
            b = float(  2 * np.dot(eta.T, np.dot(M, y)) + np.dot(A, eta)  )
            c = float(  np.dot(y.T, np.dot(M, y)) + np.dot(A, y) - off  )

            disc = b**2 - 4*a*c

            if a != 0 and disc >= 0:
                r = np.roots([a, b, c])
                interv.add( (min(r) , max(r)) , bounded = a > 0)

            elif a == 0 and b > 0:
                interv.add((-c/b, -np.inf), bounded = False)
            elif a == 0 and b < 0:
                interv.add((np.inf,  -c/b), bounded = False)
  
        return interv




def stack(quad_cons):
    quad = [con.quad_part for con in quad_cons]
    lin =  [con.lin_part  for con in quad_cons]
    off =  [con.offset    for con in quad_cons]
                      
    intersection = quad_constraints(np.vstack(quad), 
                                    np.vstack(lin),
                                    np.hstack(off))

    return intersection


    
import doctest
doctest.testmod()
