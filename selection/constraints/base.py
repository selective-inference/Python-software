import numpy as np
import mpmath as mp

from abc import ABCMeta, abstractmethod

from .intervals import intervals

from ..truncated.api import truncated_chi, truncated_chi2
from scipy.stats import norm

import quadratic

class constraint(object):
    ## Means that constraint is a Meta Class
    __metaclass__ = ABCMeta
    
    ## These two fonctions have to be defined for each subclass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def bounds(self, nu, y):
        pass

    def __invert__(self):
        inverse = cons_op.cons_op.complement(self)
        return inverse
    
    def distr_norm(self, X_s, y, sigma = 1.):
        """
        Return the value of the norm of X_s.T*y and an instance of truncated : 
        the distribution of X_s.T*y
        This is implementing the forward stepwise paper in the general case

        Parameters
        ----------
        X_s : np.float(p, k):
            X_s is a full ranked matrix

        y : np.float(p):
            y is the data, and satisfies the constraints

        sigma: float
            sigma is the variance of the normal distribution under wich
            y is chosen

        Returns
        -------
        distr : truncated_chi
            distr is an object used to study the distribution of
            np.linalg.norm(np.dot(X_s.T, y)), when y is a gaussian vector,
            chosen under the constraints and on the slice given by nu
        """
        
        p, _ = y.shape 
        # P_s = projection(X_s)

        k = min(X_s.shape)

        z = np.dot(X_s.T, y)
        z_norm = np.linalg.norm(z)

        eta = z / z_norm

        nu = np.dot(np.linalg.pinv(X_s).T, eta)
        # print "nu : ", nu
        # Computation of the intervals
        q = np.zeros((1, p, p))
        lin = (-nu).reshape((1, p))
        off = np.array([float( \
                               - np.dot(nu.T, y) \
                               + np.linalg.norm(nu)**2 * z_norm) \
                    ])

        cons_eta = quadratic.quad_constraints(q, lin, off)

        cons_inter = cons_op.intersection(self, cons_eta)

        I = cons_inter.bounds(nu, y) 
        I = I + float(z_norm)


        # Computation of theta
        Sig_s = np.dot(X_s.T, X_s)
        Sig_s_inv = np.linalg.inv(Sig_s)

        theta_s = float(sigma / np.sqrt(np.dot(eta.T, np.dot(Sig_s_inv, eta))))

        distr = truncated_chi(I._U, k, theta_s)  

        return distr

    def p_value(self, X_s, y, sigma=1.):
        """

        Examples
        --------
        The constraints are : 
        (x - 1)^2 + y ^ 2    < 4
        10 * x^2  - 10 * y^2 < -1
        (x + 1)^2 + y ^ 2    < 4

        >>> q1, lin1, off1 = np.identity(2), np.array([2., 0.]), 3.
        >>> q2, lin2, off2 = np.array([[10., 0], [0, -10.]]), np.zeros(2), -1.
        >>> q3, lin3, off3 = np.identity(2), np.array([-2., 0.]), 3.
        >>> cons = quad_constraints(np.array([q1, q2, q3]), \
                                    np.array([lin1, lin2, lin3]), \
                                    np.array([off1, off2, off3]))

        
        >>> data = cons.sample_dum(100)
        
        >>> X_s1 = np.random.multivariate_normal(np.zeros(2), \
                                                 np.identity(2), 1).T
        >>> p_values1 = [float(cons.p_value(X_s1, y)) for y in data]
        >>> #test_uniform(p_values1)
        
        >>> X_s2 = np.random.multivariate_normal(np.zeros(2), \
                                                 np.identity(2), 2).T
        >>> p_values2 = [float(cons.p_value(X_s2, y)) for y in data]
        >>> #test_uniform(p_values2)

        >>> X_s3 = np.random.multivariate_normal(np.zeros(2), \
                                                 np.identity(2), 3).T
        >>> p_values3 = [float(cons.p_value(X_s3, y)) for y in data]
        >>> #test_uniform(p_values3)
        
        """
        
        X_s = full_rank(X_s)
        k = min(X_s.shape)
        if not(self(y)):
            raise ValueError("y does not satisfies the constraints")

        distr = self.distr_norm(X_s, y, sigma)
        x = np.linalg.norm(np.dot(X_s.T, y))
        return distr.sf(x)

class cons_op(constraint):

    def __init__(self):
        self._cons_list = []
        self._op = None


    @staticmethod
    def intersection(*cons):
        if not all(isinstance(c, constraint) for c in cons):
            t = type([c for c in cons if not isinstance(c, constraint)][0])
            raise TypeError("Not a constraint : "+ repr(t))

        cons = [c for c in cons if not isinstance(c, noConstraint)]
        if all(isinstance(c, quadratic.quad_constraints) for c in cons):
            q = np.vstack([c.quad_part for c in cons])
            l = np.vstack([c.lin_part  for c in cons])
            o = np.hstack([c.offset    for c in cons])
            intersection = quadratic.quad_constraints(q, l, o)
            return intersection
        intersection = cons_op()
        intersection._cons_list = list(cons)
        intersection._op = 'I'
        return intersection


    @staticmethod
    def union(*cons):
        if not all(isinstance(c, constraint) for c in cons):
            t = type([c for c in cons if not isinstance(c, constraint)][0])
            raise TypeError("Not a constraint : "+ repr(t))
        union = cons_op()
        union._cons_list = list(cons)
        union._op = 'U'
        return union

    @staticmethod
    def complement(cons):
        if isinstance(cons, cons_op):
            op = cons._op
            if op == 'U':
                cons_gen = (~cons for cons in cons._cons_list)
                complement = cons_op.intersection(*cons_gen)
            elif op == 'I':
                cons_gen = (~cons for cons in cons._cons_list)
                complement = cons_op.union(*cons_gen)
            elif op == 'C':
                complement = cons._cons_list[0]

        else:
            complement = cons_op()
            complement._cons_list = [cons]
            complement._op = 'C'

        return complement


    def __call__(self, y):
        if self._op == 'U':
            b = any(cons(y) for cons in self._cons_list)
        elif self._op == 'I':
            b = all(cons(y) for cons in self._cons_list)
        elif self._op == 'C':
            b = not self._cons_list[0](y)
        return b
    
        
    def bounds(self, nu, y):
        if not self(y):
            raise ValueError("y does not satisfy the constraints")

        interv_gen = [cons.bounds(nu, y) for cons in self._cons_list]
            

        if self._op == 'U':
            I = intervals.union(*interv_gen)
        elif self._op == 'I':
            I = intervals.intersection(*interv_gen)
        elif self._op == 'C':
            I = ~(interv_gen[0])
        return I




class noConstraint(constraint):
    def __init__(self):
        pass

    def __call__(self, y):
        return True

    def bounds(self, nu, y):
        I = ~intervals()
        return I


