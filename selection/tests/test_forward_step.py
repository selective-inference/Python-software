import numpy as np
from ..forward_step import forward_stepwise

def test_FS():

    n, p = 100, 40
    X = np.random.standard_normal((n,p))
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_stepwise(X, Y, sigma=0.5)
    
    for i in range(30):
        FS.next()
        if not FS.check_constraints():
            raise ValueError('constraints not satisfied')

    print 'first 30 variables selected', FS.variables

    print 'M^{\pm} for the 10th selected model knowing that we performed 30 steps of forward stepwise'

    FS.model_pivots(3)
    FS.model_quadratic(3)
