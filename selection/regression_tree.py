"""
do regression tree
"""

import numpy as np
from scipy.stats import norm as ndist
from itertools import product

from .forward_step import forward_stepwise
from .binary_tree import BinaryTree

from warnings import warn

DEBUG = False

class regression_tree(BinaryTree):

    def __init__(self, X, Y, sigma=1.,
                 subset=None,
                 percentiles=np.linspace(10,90,9),
                 parent=None):
        self.n = Y.shape[0]
        self.sigma = sigma
        if subset is None:
            subset = np.ones(Y.shape[0], np.bool)
        subset_bool = np.zeros(Y.shape[0], np.bool)

        subset_bool[subset] = 1
        self.subset = subset_bool
        self.X = X[subset]
        self.percentiles = percentiles
        self.qX = np.array([np.percentile(feature, sorted(list(percentiles))) for feature in self.X.T])
        self.Y = Y[subset]
        self.splits = []
        
        BinaryTree.__init__(self, self)

    def evaluate_constraints(self, eta, values):
        """
        compute all constraints from this node down

        result will be stored in values
        """
        if hasattr(self, 'constraint'):
            V = np.dot(self.constraint, eta[self.subset])
            values.extend(V)
            self.left.evaluate_constraints(eta[self.subset], values)
            self.right.evaluate_constraints(eta[self.subset], values)
            
    def grow(self, max_depth=4):
        """
        make a split
        """
        if ((self.left is None and self.right is None)
            and self.depth < max_depth):

            bigX = []
            self.bigX_indices = []
            for i, j in product(range(self.X.shape[1]), range(self.qX.shape[1])):
                if (i,j) not in self.splits:
                    bigX.append(self.X[:,i] >= self.qX[i,j]) 
                    self.bigX_indices.append((i,j))
            bigX = np.array(bigX, np.float).reshape((-1, self.Y.shape[0])).T
            bigX -= bigX.mean(0)[None,:]
            bigX /= bigX.std(0)[None,:]
            self.bigX = bigX
            FS = iter(forward_stepwise(self.bigX, self.Y, sigma=self.sigma))
            FS.next()

            self.constraint = FS.constraints.inequality

            split = FS.variables[0]
            self.direction = self.bigX[:,split]

            i, j = self.bigX_indices[split]
            self.split_variable = int(split / self.qX.shape[1])
            self.split_quantile = split % self.qX.shape[1]
            self.splits.append((i,j))
            if DEBUG:
                print('splitting on variable %d at quantile %d' % \
                          (self.split_variable, self.split_quantile))
            split_subset = (self.X[:,i] >= self.qX[i,j]) 
            left = regression_tree(self.X, self.Y, sigma=self.sigma,
                                   subset=split_subset)
            right = regression_tree(self.X, self.Y, sigma=self.sigma,
                                    subset=~split_subset)
            self.set_left(left)
            self.left.splits.extend(self.splits)
            self.set_right(right)
            self.right.splits.extend(self.splits)
            self.left.grow(max_depth=max_depth)
            self.right.grow(max_depth=max_depth)
        elif self.depth >= max_depth:
            if DEBUG:
                print('maximum depth reached')

    def pivots(self, direction, tol=1.e-4):

        """
        A custom version of interval constraints for trees.
        """
        u = []
        Y = np.zeros(self.n)
        Y[self.subset] = self.Y
        self.evaluate_constraints(Y, u) # replaces np.dot(A, Y)
        U = np.array(u)

        if not np.all(U > -tol * np.fabs(U).max()):
            warn('constraints not satisfied: %s' % `np.min(U) / np.fabs(U).max()`)

        Sw = self.sigma**2 * direction # covariance is assumed 
                                       # to be $\sigma^2 I$
        sigma = np.sqrt((direction*Sw).sum())
        print 'sigma', sigma
        c = []; self.evaluate_constraints(Sw, c)
        C = np.array(c) / sigma**2 # replaces np.dot(A, Sw)

        V = (direction * Y).sum()
        RHS = (-U + V * C) / C
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

        # for the test, ensure that the V is positive by multiplying
        # by sign
        if V > 0:
            U = upper_bound
            L = lower_bound
            C = V
        else:
            L = -upper_bound
            U = -lower_bound
            C = -V
            
        pval = (ndist.cdf(U / sigma) - ndist.cdf(C / sigma)) / (ndist.cdf(U / sigma) - ndist.cdf(L / sigma))
        return lower_bound, V, upper_bound, sigma, pval


