import numpy as np
import quadratic_constraints
quadratic_constraints = reload(quadratic_constraints)
from quadratic_constraints import quad_constraints

import truncated_chi
truncated_chi = reload(truncated_chi)
from truncated_chi import truncated_chi

import intervals
intervals = reload(intervals)
from intervals import intervals

from time import time

from projection import projection


DOCTEST = False

class two_means(object):
    
    def __init__(self, X):
        r"""
        
        Create a new object for computing the 2-means algorithm

        Parameters
        ----------

        X : np.float((n, p))
        The n lines of X are the vector under study

        """
        self.X = X

    def _init_algorithm(self):
        r"""
        Run the first step of the 2-means algorithm
        """
        n, p = self.X.shape
        i, j = random_couple(n)
        X = self.X

        S = (np.linalg.norm(X - X[i], axis = 1) < 
             np.linalg.norm(X - X[j], axis = 1))
        self._origin = (i, j)
        self._hist = S
        self._S = S
        
        

        
    def iterate(self):
        r"""
        One iteration of the algorithm, and add the new quadratic               
        constraints
        """
        n, p = self.X.shape
        S = self._S
        X = self.X
        X_plus = X[S].mean(axis = 0)
        X_minus = X[~S].mean(axis = 0)

        S = (np.linalg.norm(X - X_plus, axis = 1) <
             np.linalg.norm(X - X_minus, axis = 1))
        
        self._hist = np.vstack((self._hist, S))
        self._S = S

    def algorithm(self, cond_stop):
        r"""
        Run the 2-means algorithm

        Parameters
        ----------
        cond_stop(n_step, W, prev_W) -> bool
        Function which decide a condition for stopping the algorithm

        Returns
        -------
        S : ndarray
        Array of bool, splitting the data points into two clusters
        """

        X = self.X
        n, p = X.shape

        self._W = X.var(axis = 0).sum()
        self._init_algorithm()

        n_step = 0
        while n_step == 0 or cond_stop(n_step, self._W, prev_W):
            prev_W = self._W
            
            self.iterate()
            S = self._S
            N_1 = S.sum()
            self._W = N_1 * X[S].var(axis = 0).sum() + \
                      (n - N_1) * X[~S].var(axis = 0).sum()

            n_step += 1
        


    def constraints_path(self):
        r"""
        Return the constraints described by the path of the algorithm

        Returns
        -------
        constraints : quad_constraints
              Contains the constraints given by all the hyperplan, the 
              objective function which is decreasing and the linear 
              equality for the eta

        """
        t = time()
        n, p = self.X.shape

        cons = []
        
        hist = self._hist[1:]
        neg_hist = ~self._hist[1:]
        n_step = len(self._hist)

        ## The inequalities given by the memberships in the 
        ## hyperplan after the first step
        for i in range(1, len(self._hist)):
            S = self._hist[i]
            last_S = self._hist[i-1]


            last_N1 = last_S.sum()
            last_N2 = n - last_N1

            alpha = np.identity(n) \
                    - 1./last_N1 * np.outer( S,  last_S) \ 
                    - 1./last_N2 * np.outer(~S, ~last_S) 

            beta = np.identity(n) \
                   - 1./last_N2 * np.outer( S, ~last_S) \
                   - 1./last_N1 * np.outer(~S,  last_S)

            Q_array = np.einsum('ij, ik -> ijk', alpha, alpha) - \
                      np.einsum('ij, ik -> ijk', beta , beta )

            Q_tensor = np.einsum('abc, de -> abdce', Q_array, np.identity(p))
            Q_tensor = Q_tensor.reshape((n, n*p, n*p))
            
            cons.append( Q_tensor )
        
        arr_cons = np.vstack(cons)

        ## The inequalities given by the first  step
        S = self._hist[0]
        i, j = self._origin
        Ei = Ej = np.zeros(n)
        Ei[i] = Ei[j] = 1.
        alpha = np.identity(n) - np.outer( S, Ei) - np.outer(~S, Ej)
        beta  = np.identity(n) - np.outer(~S, Ei) - np.outer( S, Ej)

        Q_array = np.einsum('ij, ik -> ijk', alpha, alpha) - \
                  np.einsum('ij, ik -> ijk', beta , beta )

        Q_tensor = np.einsum('abc, de -> abdce', Q_array, np.identity(p))
        Q_tensor = Q_tensor.reshape((n, n*p, n*p))

        arr_cons = np.vstack((arr_cons, Q_tensor))

        

        ## The inequalities given by the objective function which
        ## is decreasing 
        hist = self._hist

        P1 = np.einsum('ij, ik -> ijk',  hist,  hist)
        N1_hist =  (hist.sum(axis=1).reshape((n_step, 1, 1))).astype(np.float)
        P1 = P1/N1_hist

        P2 = np.einsum('ij, ik -> ijk', ~hist, ~hist)
        N2_hist = ((~hist).sum(axis=1).reshape((n_step, 1, 1))).astype(np.float)
        P2 = P2/N2_hist

        P = np.identity(n) - P1 - P2

        Q = np.einsum('akb, akc -> abc', P, P)

        Q_array = Q[1:] - Q[:-1]

        Q_tensor = np.einsum('abc, de -> abdce', Q_array, np.identity(p))
        Q_tensor = Q_tensor.reshape((n_step - 1, n*p, n*p))

        arr_cons = np.vstack((arr_cons, Q_tensor))

        
        ## The linear inequality given by eta
        S = self._S.reshape((1, n))

        u = S/float(S.sum()) - (~S)/float(~S.sum())

        eta = np.dot(u.T, np.dot(u, self.X))
        eta = eta.reshape(n*p,)
        eta = eta/np.linalg.norm(eta)

        cons_lin = quad_constraints([np.zeros((n*p, n*p))], np.array([-eta]))
        
            
        ## End of making constraints
        constraints = quad_constraints(arr_cons)
        constraints = quadratic_constraints.stack((constraints, cons_lin))
        return constraints

    def bounds_test(self):
        """
        Gives the intervals selectionned

        """
        n, p = self.X.shape
        X_asvector = self.X.reshape((n*p, 1))
        S = self._S.reshape((1, n))

        u = S/float(S.sum()) - (~S)/float(~S.sum())

        eta = np.dot(u.T, np.dot(u, self.X))
        eta = eta.reshape(n*p, 1)
        eta = eta/np.linalg.norm(eta)

        cons = self.constraints_path()


        I = cons.bounds(eta, X_asvector)
       
        I.offset(float(np.dot(eta.T, X_asvector)))
        return I

        
    def p_val(self):
        n, p = self.X.shape
        X_asvector = self.X.reshape((n*p,1))
        S = self._S.reshape((1, n))

        u = S/float(S.sum()) - (~S)/float((~S).sum())

        eta = np.dot(u.T, np.dot(u, self.X))
        eta = eta.reshape(n*p, 1)
        eta = eta/np.linalg.norm(eta)

        distr = self.test_distribution()

        x = np.dot(eta.T, X_asvector)
        p = distr.cdf(x)

        #print x, p
        return p
        

    def test_distribution(self, sigma=1):
        n, p = self.X.shape
        X_asvector = self.X.reshape((n*p,))
        S = self._S

        N_1, N_2 = S.sum(), (~S).sum()

        a_s = 1./np.sqrt(n) * ( np.sqrt(N_2/N_1) * S + np.sqrt(N_1/N_2) * (~S) )
        #M_s is the X_s.T  of the forward stepwise
        M_s = tensor_reshape(a_s, np.identity(p))
        P_s = projection(M_s.T)
    
        theta_s = sigma * np.dot(X_asvector, P_s(X_asvector))
        theta_s /= np.linalg.norm(np.dot(M_s, X_asvector))**2

        I = self.bounds_test()

        tr_chi = truncated_chi(I.intersection(), p, theta_s)
        #print I.intersection()

        return tr_chi



def random_couple(n):
    """
    Return a random couple of integers (i,j) such as i < n and j < n
    and i not equal to j, uniformaly
    """
    i = np.random.randint(n)
    j = np.random.randint(n-1)
    if j >= i:
        j += 1
    return i,j


def tensor_reshape(a, b):
    if a.ndim == 1:
        a = np.array([a])
    
    tensor = np.einsum('ab, cd -> acbd', a, b)
    
    l1, l2 = b.shape
    L1, L2 = a.shape

    return tensor.reshape((L1*l1, L2*l2))
    



def sample_gaussians(n_points, param):
    """
    param: array of (mean, Sigma)
    """
    k = len(param)
    n_points_by_cluster = np.random.multinomial(n_points, [1./k]*k)
    
    t = [np.random.multivariate_normal(mean, cov, size = n_i) 
         for ((mean, cov), n_i) in zip(param, n_points_by_cluster)]

    sample = np.vstack(t)

    return sample, n_points_by_cluster


def cond_stop(n_step, sigma, last_sig):
    return n_step < 150 and sigma != last_sig

p_array = []
for i in range(50):
    if i%10 == 0:
        print i
    gauss1 = [np.zeros(5), np.identity(5)]
    # gauss2 = [np.array([4., 0.]), np.identity(2)]
    # param =  [gauss1, gauss2]
    
    sample, t = sample_gaussians(50, [gauss1])
    X = sample.copy()
    t_m = two_means(sample)         
    
    t_m.algorithm(cond_stop)
    
    p = float(t_m.p_val())

    p_array.append(p)

p_array = sorted(p_array)

import matplotlib.pyplot as plt

x = np.arange(0, 1, 1./len(p_array));
plt.plot(x, p_array)





def plot(X1, X2, S):
    import matplotlib
    import matplotlib.pyplot as plt

    X = np.vstack((X1, X2))
    fig, ax = plt.subplots()
        
    c1 = np.dot(S,  X)/sum( S)
    c2 = np.dot(~S, X)/sum(~S)
    
    bis1, bis2 = bisection(c1, c2)
    ax.plot(bis1, bis2, 'k--')


    ax.plot(c1[0], c1[1], 'gx')
    ax.plot(c2[0], c2[1], 'gx')
    ax.scatter(X1[:,0], X1[:,1], c='r')
    ax.scatter(X2[:,0], X2[:,1], c='b')

    plt.show()

def bisection(c1, c2):
    mean = (c1+c2)/2
    c1, c2 = c1 - mean, c2 - mean
    c1 = np.array([-c1[1], c1[0]]) + mean
    c2 = np.array([-c2[1], c2[0]]) + mean
    return c1, c2
    

#plot(X[:t[0]], X[t[0]:], t_m._S)








if DOCTEST:
    import doctest
    doctest.testmod()



