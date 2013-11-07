import numpy as np
from selection.regression_tree import regression_tree

def test_tree():

    n, p = 500, 40
    X = np.random.standard_normal((n,p))
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(n) * 0.5
    Y += 6 * (X[:,10] > np.median(X[:,10]))
    Y -= Y.mean()
    
    tree = regression_tree(X, Y, sigma=0.5, 
                           subset=range(495))
    print tree.X.shape
    tree.grow(max_depth=3)
    print tree.split_variable, tree.split_quantile
    cvals = []
    D = np.zeros(n)
    D[tree.subset] = tree.direction
    tree.evaluate_constraints(D, cvals)
    print tree.pivots(np.random.standard_normal(n))
    A = np.array(cvals)
    print A.shape
    return tree

def test_tree_small():

    n, p = 500, 2
    X = np.random.standard_normal((n,p))
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(n) * 0.5
    Y += 6 * (X[:,0] > np.median(X[:,0]))
    Y -= Y.mean()
    
    tree = regression_tree(X, Y, sigma=0.5, 
                           subset=range(495))
    print tree.X.shape
    tree.grow(max_depth=3)
    print tree.split_variable, tree.split_quantile
    cvals = []
    D = np.zeros(n)
    D[tree.subset] = tree.direction
    tree.evaluate_constraints(D, cvals)
    print tree.pivots(np.random.standard_normal(n))
    A = np.array(cvals)
    print A.shape
    return tree

if __name__ == "__main__":
    tree = test_tree()
