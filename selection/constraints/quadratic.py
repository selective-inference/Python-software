from __future__ import print_function
import numpy as np
import mpmath as mp
from scipy.stats import norm

from ..truncated.api import truncated_chi, truncated_chi2, truncated_F
from .intervals import intervals
from .base import constraints as base_constraints

class constraints(base_constraints):

    r"""

    Object solving the problems of slices for quadratic and linear 
    inequalities

    .. math::
    
          \forall i, y^T Q_i y + a_i^t < b_i

    """
    
    def __init__(self, 
                 quad_part, 
                 lin_part=None, 
                 offset=None,
                 covariance=None,
                 mean=None,
                 rank=None):
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

        covariance : np.float((p,p))
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : np.float(p)
            Mean vector of Gaussian distribution to be 
            truncated. Defaults to `np.zeros(self.dim)`.

        rank : int
            If not None, this should specify
            the rank of the covariance matrix. Defaults
            to self.dim.

        WARNING : The shapes of the three parameters must fit
        """
              
        # Check the inputs are aligned
        p, _ = quad_part[0].shape
        self.dim = p
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
                

        self.quad_part = np.array(quad_part)
        self.lin_part = np.array(lin_part)
        
        self.offset = offset

        if rank is None:
            self.rank = self.dim
        else:
            self.rank = rank

        if covariance is None:
            covariance = np.identity(self.dim)
        self.covariance = covariance

        if mean is None:
            mean = np.zeros(self.dim)
        self.mean = mean

    def __call__(self, y, tol=1.e-10):
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

        Examples
        --------
        The constraints are : 
        (x + 1)^2 + y ^ 2    < 4
        10 * x^2  - 10 * y^2 < -1
        (x - 1)^2 + y ^ 2    < 4

        >>> q1, lin1, off1 = np.identity(2), np.array([2., 0.]), 3.
        >>> q2, lin2, off2 = np.array([[10., 0], [0, -10.]]), np.zeros(2), -1.
        >>> q3, lin3, off3 = np.identity(2), np.array([-2., 0.]), 3.
        >>> cons = constraints(np.array([q1, q2, q3]), \
                               np.array([lin1, lin2, lin3]), \
                               np.array([off1, off2, off3]))

        >>> y1 = np.array([[0. , 1. ]]).T
        >>> y2 = np.array([[0. , -1.]]).T
        >>> y3 = np.array([[.5 , 1. ]]).T
        >>> y4 = np.array([[0. , 0. ]]).T
        >>> y5 = np.array([[.5 , 1.5]]).T
        >>> y6 = np.array([[2.,  0. ]]).T
        >>> cons(y1), cons(y2), cons(y3), cons(y4), cons(y5), cons(y6)
        (True, True, True, False, False, False)
        """
        V1 =  np.einsum('bi, abc, cj -> aij', y, self.quad_part, y).reshape(-1) \
              + np.dot(self.lin_part, y).reshape(-1) \
              - self.offset
        return np.all(V1 < tol * np.linalg.norm(V1, ord = np.inf))



    # def append(self, q, lin, off):
    #     """
    #     Add an inequality to the set of constraints : 
    #     y.T * q * y + lin * y < off


    #     """
    #     p = self.quad_part.shape[1]
    #     if q.shape[0] != p:
    #         raise ValueError("Matrix are not aligned")
        
    #     self.quad_part = np.vstack((self.quad_part, q.reshape((1, p, p)) ))
    #     self.lin_part  = np.vstack((self.lin_part , lin.reshape((1, p)) ))
    #     self.offset    = np.hstack((self.offset   , off  ))

        
    # def pop(self):
    #     """
    #     Delete the last inequality
    #     """
    #     q = self.quad_part[-1]
    #     lin = self.lin_part[-1]
    #     off = self.offset[-1]

    #     self.quad_part = self.quad_part[:-1]
    #     self.lin_part  = self.lin_part[:-1]
    #     self.offset    = self.offset[:-1]

    #     return q, lin, off


        
    def bounds(self, nu, y):
        """
        Return the intervals of the slice in a direction nu, which 
        respects the inequality

        Parameters
        ----------
        
        nu : np.float(p)
              The direction of the slice

        y : np.float(p)
              A point on the affine slice

        Returns
        -------
        intervals : array of couples
              Array of (a, b), which means that the set is the union
              of [a, b]

        Examples
        --------
        The constraints are : 
        (x - 1)^2 + y ^ 2    < 4
        10 * x^2  - 10 * y^2 < -1
        (x + 1)^2 + y ^ 2    < 4

        >>> q1, lin1, off1 = np.identity(2), np.array([2., 0.]), 3.
        >>> q2, lin2, off2 = np.array([[10., 0], [0, -10.]]), np.zeros(2), -1.
        >>> q3, lin3, off3 = np.identity(2), np.array([-2., 0.]), 3.
        >>> cons = constraints(np.array([q1, q2, q3]), \
                               np.array([lin1, lin2, lin3]), \
                               np.array([off1, off2, off3]))

        >>> y = np.array([[0., -1.]]).T

        >>> nu1 = np.array([[0., 1.]]).T
        >>> I1 = np.array(cons.bounds(nu1, y).intersection())
        >>> I1_expected = np.array([[- np.sqrt(3) + 1, 1. -1./np.sqrt(10)], \
                                     [1. + 1./np.sqrt(10), 1. + np.sqrt(3)]])
        >>> #np.all(np.fabs(I1 - I1_expected) < 1.e-10)

        >>> nu2 = 2*nu1             ## Try with a nu not unitary
        >>> I2 = np.array(cons.bounds(nu2, y).intersection())
        >>> I2_expected = I1_expected /2
        >>> #np.all(np.fabs(I2 - I2_expected) < 1.e-10)

        >>> nu3 = np.array([[1., 0]]).T
        >>> I3 = np.array(cons.bounds(nu3, y).intersection())
        >>> I3_expected = np.array([[1. - np.sqrt(3), -1. + np.sqrt(3)]])
        >>> #np.all(np.fabs(I3 - I3_expected) < 1.e-10)
        

        """
        
        if not self(y):
            raise ValueError('y does not respect the constraints')

        interv_list = []

        for M, A, off in zip(self.quad_part, self.lin_part, self.offset):

            mp.dps = 15
            a = float(  np.dot(nu.T, np.dot(M, nu))  )
            b = float(  2 * np.dot(nu.T, np.dot(M, y)) + np.dot(A, nu)  )
            c = float(np.dot(y.T, np.dot(M, y)) + np.dot(A, y) - off )
            
            
            if c > 0:
                raise ValueError("c should be negative : " + repr(c))

            disc = b**2 - 4*a*c
            
            if a != 0 and disc >= 0:

                r = roots_poly2(a, b, c)
                
                I = intervals((float(min(r)), float(max(r))))
                interv_list.append(I if a > 0 else ~I)

            elif a == 0 and b != 0:
                I = intervals((-np.inf, -c/b))
                interv_list.append(I if b > 0 else ~I)
            
            if (disc > 0 or (a== 0 and b != 0)) and not interv_list[-1](0):
                raise ValueError("0 is not in the interval")
         
        interv = intervals.intersection(*interv_list)

        return interv

    def bounds_unknownSigma(self, P, y):
        n = len(y)
        I = np.identity(n)
        U1 = np.dot(P, y)
        U2 = np.dot(I-P, y)
        U1 = U1/np.linalg.norm(U1)
        U2 = U2/np.linalg.norm(U2)

        interv_list = []

        for M, A, off in zip(self.quad_part, self.lin_part, self.offset):

            a = float(  np.dot(U1.T, np.dot(M, U1))  )
            b = float(  2 * np.dot(U2.T, np.dot(M, U1)) )
            c = float(  np.dot(U2.T, np.dot(M, U2))  )
            
            disc = b**2 - 4*a*c
            
            if a != 0 and disc >= 0:

                r = roots_poly2(a, b, c)
                
                I = intervals((float(min(r)), float(max(r))))
                interv_list.append(I if a > 0 else ~I)

            elif a == 0 and b != 0:
                I = intervals((-np.inf, -c/b))
                interv_list.append(I if b > 0 else ~I)
            
         
        interv = intervals.intersection(*interv_list)

        return interv
        


    

    def sample(self, n_sample, initial_point):
        initial_point = initial_point.reshape(-1)
        quad = self.quad_part
        quad_lin = self.lin_part
        offset_quad = self.offset.reshape(-1)
        lin = np.array([]).reshape((0,0))
        offset_lin = np.array([])

        # this import does not exist
        samples = quad_sampler(n_sample,
                               initial_point,
                               quad,
                               quad_lin,
                               lin,
                               offset_quad,
                               offset_lin)

        samples = [v.reshape((len(v), 1)) for v in samples]
        
        if not all(self(v) for v in samples):
            raise ValueError("The samples are not correct")

        return samples

    def sample_dum(self, n_sample):
        p = self.quad_part.shape[1]

        samples = []
        while len(samples) < n_sample:
            y = np.random.multivariate_normal(np.zeros(p), \
                                              np.identity(p)).reshape((p, 1))
            if self(y):
                samples.append(y)
        return samples


    def gen_p_value(self, y, f, sigma=1., n_samples=1000):
        """
        Parameters
        ----------
        f : z -> x
            observed value
        """
        samples = self.sample(n_samples, y)
        observed_values = np.sort([f(z) for z in samples])
        obs = f(y)
        k = max([i for i in range(n_samples) if observed_values[i] < obs])

        return float(k)/n_samples
        
    def distr_norm_unknownSigma(self, P, y, n, p):

        I = self.bounds_unknownSigma(P, y) 


        # Computation of theta
        theta_s = float(2)/float(n-2)

        distr = truncated_F(I._U, n*p - 2*p, 2*p, theta_s)  

        return distr

    def p_value_unknownSigma(self, P, y, n, p):
        distr = self.distr_norm_unknownSigma(P, y, n, p)
        x = (np.linalg.norm(np.dot(P, y))/ \
             np.linalg.norm(np.dot(np.identity(n*p) - P, y)))**2
        return distr.sf(x)

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

        cons_eta = constraints(q, lin, off)

        cons_inter = stack(self, cons_eta)

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

       
        """
        #X_s = full_rank(X_s) -- not sure what this function was
        k = min(X_s.shape)
        if not(self(y)):
            raise ValueError("y does not satisfies the constraints")

        distr = self.distr_norm(X_s, y, sigma)
        x = np.linalg.norm(np.dot(X_s.T, y))
        return distr.sf(x)


class constraints_vecnorm(constraints):
    def __init__(self, alpha, beta, dim):
        self._alpha = alpha
        self._beta  = beta
        self._dim   = dim

    def __call__(self, y, tol=1.e-10):
        X = y.reshape(self._dim)
        A = np.dot(self._alpha, X)
        B = np.dot(self._beta , X)
        V1 = np.linalg.norm(A, axis = 1) - np.linalg.norm(B, axis = 1)
        return np.all(V1 < tol * np.linalg.norm(V1, ord = np.inf))

    def bounds(self, nu, y):
        if not self(y):
            raise ValueError('y does not respect the constraints')

        X = y.reshape(self._dim)
        nu = nu.reshape(self._dim)

        interv_list = []
        U1 = np.dot(self._alpha, X )
        U2 = np.dot(self._beta , X )
        V1 = np.dot(self._alpha, nu)
        V2 = np.dot(self._beta,  nu)


        for u1, u2, v1, v2 in zip(U1, U2, V1, V2):
            a = np.linalg.norm(v1)**2 - np.linalg.norm(v2)**2
            b = 2* float( np.dot(u1, v1.T) - np.dot(u2, v2.T) )
            c = np.linalg.norm(u1)**2 - np.linalg.norm(u2)**2
            
            if c > 0:
                raise ValueError("c should be negative : " + repr(c))

            disc = b**2 - 4*a*c
            
            if a != 0 and disc >= 0:

                r = roots_poly2(a, b, c)
                
                I = intervals((float(min(r)), float(max(r))))
                interv_list.append(I if a > 0 else ~I)

            elif a == 0 and b != 0:
                I = intervals((-np.inf, -c/b))
                interv_list.append(I if b > 0 else ~I)
            
            if (disc > 0 or (a== 0 and b != 0)) and not interv_list[-1](0):
                raise ValueError("l'intervalle ne contient pas 0 : " + repr(interv_list[-1]))
         
        interv = intervals.intersection(*interv_list)

        return interv
        

def stack(*quad_cons):
    quad = [con.quad_part for con in quad_cons]
    lin =  [con.lin_part  for con in quad_cons]
    off =  [con.offset    for con in quad_cons]
                      
    intersection = constraints(np.vstack(quad), 
                               np.vstack(lin),
                               np.hstack(off))

    return intersection



def roots_poly2(a, b, c):
    disc = b**2 - 4*a*c

    if disc < 0:
        raise ValueError("The polynom has no roots")

    
    r = (-b + np.sqrt(disc) * np.array([1., -1.])) / (2. * a)
    r1 = r[np.argmax(np.fabs(r))]
    if c == 0:
        r2 = 0.
    else:
        r2 = c / (r1*a)
    return (r1, r2)
    



if __name__ == "__main__":
    import doctest
    doctest.testmod()

