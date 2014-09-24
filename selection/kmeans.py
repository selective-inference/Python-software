import numpy as np
import random

from quadratic_constraints import quad_constraints, quad_constraints_vecnorm
from constraint import constraint, cons_op, noConstraint
from truncated import truncated_chi
from intervals import intervals
from projection import projection

import warnings

from tools import timethis

#from time import time


class kmeans(object):
    
    def __init__(self, X, k):
        r"""
        
        Create a new object for computing the 2-means algorithm

        Parameters
        ----------

        X : np.float((n, p))
        The n lines of X are the vector under study

        """
        self._X = X
        self._K = k

    def _init_algorithm(self, init_points):
        r"""
        Run the first step of the k-means algorithm
        """
        X = self._X
        n, p = X.shape
        K = self._K
            
        Centroids = np.array([X[k] for k in init_points])
        
        Dist = np.array([[np.linalg.norm(X[i] - Centroids[k]) for k in range(K)]
                         for i in range(n)])
        #print "Dist init : ", Dist
        S = np.argmin(Dist, axis=1)
        #print S

        return S
        
        

        
    def _iterate(self, S):
        r"""
        One iteration of the algorithm, and add the new quadratic               
        constraints
        """
        X = self._X
        n, p = X.shape
        K = self._K

        try:
            Centroids = np.array([X[ S == k ].mean(axis=0) for k in range(K) ])
        except Warning:
            raise ValueError("Centroid disapeared")
        Dist = np.array([[np.linalg.norm(X[i] - Centroids[k]) for k in range(K)]
                         for i in range(n)])


        S = np.argmin(Dist, axis = 1)

        return S

    @timethis
    def algorithm(self, cond_stop=None, init_points_list=None):
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

        X = self._X
        n, p = X.shape
        K = self._K

        if init_points_list == None:
            init_points_list = [random_tuple(K, n)]
        self._init_points_array = np.array(init_points_list)

        hist_array = []
        w_array = []
        #plot_X(X)

        for init_points in init_points_list:
            S = self._init_algorithm(init_points)
            hist = S

            n_step = 0
            while n_step < 2 or not np.all(hist[-1] == hist[-2]):
                #print n_step
                S = self._iterate(S)
                hist = np.vstack((hist, S))

                n_step += 1

            hist_array.append(hist)
            w = self.objective(S)
            w_array.append(w)
        self._hist = hist_array

        i_min = np.argmin(w_array)
        S_min = self._hist[i_min][-1]

        self._S = S_min
        return S_min




    def objective(self, S):
        X = self._X
        K = self._K

        Centroids = np.array([X[ S == k ].mean(axis=0) for k in range(K) ])

        w = sum( (S == k).sum() * np.linalg.norm( X[S == k] - Centroids[k] ) \
                 for k in range(K))

        return w
        

    @timethis
    def _constraints_path(self, path, init_points, version):
        r"""
        Return the constraints described by the path of the algorithm

        Returns
        -------
        constraints : quad_constraints
              Contains the constraints given by all the hyperplan, the 
              objective function which is decreasing and the linear 
              equality for the eta

        """
        n, p = self._X.shape
        X = self._X
        K = self._K
        X_asvector = X.reshape((n*p,1))

        cons = []
        
        n_step = len(path)

        #version = 1

        alpha_array = []
        beta_array  = []



        for i in range(1, n_step):
            
            S = np.array([(path[i]==k) for k in range(K)])
            last_S = np.array([(path[i-1]==k) for k in range(K)])
            
            last_N = last_S.sum(axis=1).reshape((K, 1))
            last_S = 1./last_N * last_S
            
            last_S_rolled = np.array([np.roll(last_S, k, axis=0) \
                                      for k in range(K)])
            
            M = np.identity(n) - np.einsum('ab, cad -> cbd', S, last_S_rolled)
            
            alpha = np.vstack([M[0] for k in range(K-1)])
            beta =  np.vstack(M[1:])
            
            alpha_array.append(alpha)
            beta_array.append(beta)

        
        ## The inequalities given by the first  step


        S = np.array([(path[0]==k) for k in range(K)])
        last_S = np.zeros((K, n))
        for k in range(K):
            last_S[k, init_points[k]] = 1.
            
        last_S_rolled = np.array([np.roll(last_S, k, axis=0) \
                                  for k in range(K)])

        M = np.identity(n) - np.einsum('ab, cad -> cbd', S, last_S_rolled)
            
        alpha = np.vstack([M[0] for k in range(K-1)])
        beta =  np.vstack(M[1:])
            
        alpha_array.append(alpha)
        beta_array.append(beta)
            
        alpha_cons = np.vstack(alpha_array)
        beta_cons = np.vstack(beta_array)

        ## End of making constraints            
        constraints = quad_constraints_vecnorm(alpha_cons, beta_cons, (n, p))
        
        return constraints

    @timethis
    def _constraints_minset(self):
        X = self._X
        n, p = X.shape
        K = self._K
        hist = self._hist
        S_min = self._S
    

        other_S = [h[-1] for h in hist if not equivalent_set(h[-1], S_min, K)]
        
        if len(other_S) == 0:
            return noConstraint()

        Q_array = []
        alpha = np.array([(S_min==k) for k in range(K)])
        Q_a = - np.einsum('ij, ik -> jk', alpha, alpha)

        N = np.array([(S_min==k) for k in range(K)]).sum(axis=1)
        D = np.array([N[S_min[i]] for i in range(n)])
        Q_diag = np.diag(D)
        Q_a = Q_diag + Q_a
        
        for S in other_S:
            beta = np.array([(S == k) for k in range(K)])
            Q_b = np.einsum('ij, ik -> jk', beta, beta)

            N = np.array([(S==k) for k in range(K)]).sum(axis=1)
            D = np.array([N[S[i]] for i in range(n)])
            Q_diag = np.diag(D)
            Q_b = Q_diag + Q_b

            Q_array.append((Q_a - Q_b).reshape((1, n,n)))
        Q_array = np.vstack(Q_array)
        
        Q_tensor = np.einsum('abc, de -> abdce', Q_array, np.identity(p))
        Q_tensor = Q_tensor.reshape((len(other_S), n*p, n*p))

        cons = quad_constraints(Q_tensor)
        return cons

    def constraints_algo(self, version):
        init_points_array = self._init_points_array
        hist = self._hist
        cons_gen = [self._constraints_path(hist[i], init_points_array[i], version) 
                    for i in range(len(init_points_array))]
        
        cons = cons_op.intersection(*cons_gen)

        cons_minset = self._constraints_minset()

        cons = cons_op.intersection(cons, cons_minset)
        return cons


    @timethis
    def p_val(self):
        X = self._X
        n, p = X.shape
        X_asvector = X.reshape((n*p,1))
        S = self._S
        K = self._K

        cons = self.constraints_algo()


        alpha = np.array([(S==k) for k in range(K)])
        M_s = np.identity(n) - np.einsum('ij, ik -> jk', alpha, alpha)

        M_s = np.einsum('ab, cd -> acbd', M_s, np.identity(p)).reshape((n*p, n*p))

        p = cons.p_value(M_s, X_asvector, 1.)

        return p

    def p_val_unknownSigma(self):
        X = self._X
        n, p = X.shape
        X_asvector = X.reshape((n*p,1))
        S = self._S
        K = self._K

        cons = self.constraints_algo()

        alpha = np.array([(S==k) for k in range(K)])
        M_s = np.identity(n) - np.einsum('ij, ik -> jk', alpha, alpha)
        M_s = np.einsum('ab, cd -> acbd', M_s, np.identity(p)).reshape((n*p, n*p))

        p = cons.p_value_unknownSigma(M_s, X_asvector, n, p)
        #print "Voici la p-valeur : ", p

        return p

    def p_val_sample(self):
        X = self._X
        n, p = X.shape
        X_asvector = X.reshape((n*p,1))
        S = self._S
        K = self._K

        cons = self.constraints_algo()

        # cons0 = cons._cons_list[0]
        # cons1 = cons._cons_list[1]
        # q1, l1, o1 = cons0.quad_part, cons0.lin_part, cons0.offset
        # q2, l2, o2 = cons1.quad_part, cons1.lin_part, cons0.offset
        # q, l, o = np.vstack((q1, q2)), np.vstack((l1, l2)), np.hstack((o1, o2))
        # cons = quad_constraints(q, l, o)

        alpha = np.array([(S==k) for k in range(K)])
        M_s = np.identity(n) - np.einsum('ij, ik -> jk', alpha, alpha)

        M_s = np.einsum('ab, cd -> acbd', M_s, np.identity(p)).reshape((n*p, n*p))

        def value_observed(y):
            return np.linalg.norm(np.dot(M_s, y))
        
        p = cons.gen_p_value(X_asvector, value_observed)

        return p


def random_tuple(k, n):
    """
    Return a random tuple of k integers lower than n, all differents, uniformaly              
    """
    t = random.sample([x for x in range(n)], k)
    return tuple(t)



@timethis
def equivalent_set(S1, S2, K):
    n = len(S1)
    perm = np.array([-1 for x in range(K)])
    i = 0
    b = True
    while b and i < n:
        if perm[S2[i]] == -1:
            perm[S2[i]] = S1[i]
        else:
            b = perm[S2[i]] == S1[i]
        i += 1
    return b



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
    #n_points_by_cluster = np.random.multinomial(n_points, [1./k]*k)
    n_points_by_cluster = [n_points/2, n_points/2]
    t = [np.random.multivariate_normal(mean, cov, size = n_i) 
         for ((mean, cov), n_i) in zip(param, n_points_by_cluster)]             

    sample = np.vstack(t)

    return sample, n_points_by_cluster


def cond_stop(n_step, sigma, last_sig):
    return n_step < 150 and sigma != last_sig

@timethis
def f(n, p, k, n_initial_points, dist = 0, sample_bool=False):
    n_sample = 1
    p_array = []
    direction = np.random.multivariate_normal(np.zeros(p), np.identity(p))
    direction = direction/np.linalg.norm(direction)
    print "direction : ", direction
    gauss1 = [ dist * direction/2, np.identity(p)]
    gauss2 = [ dist *-direction/2, np.identity(p)]

    # gauss1[0][0] = float(dist)/2
    # gauss2[0][0] = - float(dist)/2

    param = [gauss1, gauss2]
    while len(p_array) < n_sample:
        if len(p_array) % 50 == 0:
            print len(p_array)
        sample, t = sample_gaussians(n, param)
        t_m = kmeans(sample, k)  

        # X = t_m._X
        # n, p = X.shape
        # X_asvector = X.reshape((n*p,1))

        init_points_list = [random_tuple(k,n) for x in range(n_initial_points)]

        try:
            t_m.algorithm(init_points_list=init_points_list)
        except:
            print "bug"
            raise
        # unknown = False
        # if not sample_bool and not unknown:
        #     p_value = float(t_m.p_val())
        # elif unknown:
        #     p_value = float(t_m.p_val_unknownSigma())
        # else:
        #     p_value = float(t_m.p_val_sample())
        # print p_value
        # p_array.append(p_value)
        p_array.append(0)
        c1 = t_m.constraints_algo(1)
        c2 = t_m.constraints_algo(2)
        
    #return p_array
    return c1, c2, t_m



# p_array = f(50, 5, 0)           
# print f(4, 5, 10)

# import matplotlib.pyplot as plt

# x = np.arange(0, 1, 1./len(p_array));
# plt.plot(x, p_array, 'ro')


def plot_X(X):
    import matplotlib
    import matplotlib.pyplot as plt

    #X = np.array([X])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    #plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    
    print "plot_X", X.shape

    plt.scatter(X[:, 0], X[:, 1] , s=30, c='r', marker = 'o', alpha=1.)

    plt.show()



def plot(X, S, K, limit):
    import matplotlib
    import matplotlib.pyplot as plt

    colors = ['r', 'g', 'c', 'm', 'y', 'k', 'w']

    #X = np.vstack((X1, X2))
    Centroids = np.array([X[ S == k ].mean(axis=0) for k in range(K) ])

    #fig, ax = plt.subplots()
    if limit == 0:
        X = np.array([X])
    else:
        X = np.array([X[:limit], X[limit:]])
   
    points = Centroids
    if points.shape[0] < 3:
        points = np.vstack((points, np.array([0., 100.])))

    # compute Voronoi tesselation
    vor = Voronoi(points)
    
    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    
    # colorize
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.2)
        
    

    #plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    #plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    

    for x, c in zip(X, colors):
        #print "x :", x
        plt.scatter(x[:, 0], x[:, 1], s=30, c= c, marker='o', alpha=1.)

    plt.scatter(points[:,0], points[:,1], s=40, color='k', marker = 'v')
    plt.show()

def bisection(c1, c2):
    mean = (c1+c2)/2
    c1, c2 = c1 - mean, c2 - mean
    c1 = np.array([-c1[1], c1[0]]) + mean
    c2 = np.array([-c2[1], c2[0]]) + mean
    return c1, c2
    

#plot(X[:t[0]], X[t[0]:], t_m._S)



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi


## Code found in : http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = 10* vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



if __name__ == "__main__":
    import doctest
    doctest.testmod()



