import numpy as np
from selection.forward_step import forward_stepwise
import matplotlib.pyplot as plt
import statsmodels.api as sm

def test_FS():

    n, p = 100, 40
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
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

def simulate_null():

    n, p = 100, 40
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_stepwise(X, Y, sigma=0.5)
    
    for i in range(5):
        FS.next()

    return [p[-1] for p in FS.model_pivots(3)]

def test_ecdf(nsim=1000):
    
    P = []
    for _ in range(nsim):
        P.append(simulate_null())
    P = np.array(P)

    ecdf1 = sm.distributions.ECDF(P[:,0])
    ecdf2 = sm.distributions.ECDF(P[:,1])
    ecdf3 = sm.distributions.ECDF(P[:,2])

    plt.clf()
    plt.plot(ecdf1.x, ecdf1.y, linewidth=4, color='black', label='Fixed (but random) $A$')
    plt.plot(ecdf2.x, ecdf2.y, linewidth=4, color='purple', label='Selected $A$')
    plt.plot(ecdf3.x, ecdf3.y, linewidth=4, color='green', label='Deterministic $A$')
    plt.show()
