import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from selection.algorithms.forward_step import forward_stepwise, info_crit_stop, sequential

def test_FS(k=10):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_stepwise(X, Y, covariance=0.5**2 * np.identity(n))

    for i in range(k):
        FS.next()

    print 'first %s variables selected' % k, FS.variables

    print 'pivots for 3rd selected model knowing that we performed %d steps of forward stepwise' % k

    print FS.model_pivots(3)
    print FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=5000, ndraw=5000)
    print FS.model_quadratic(3)

def test_FS_unknown(k=10):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_stepwise(X, Y)

    for i in range(k):
        FS.next()

    print 'first %s variables selected' % k, FS.variables

    print 'pivots for last variable of 3rd selected model knowing that we performed %d steps of forward stepwise' % k

    print FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=5000, ndraw=5000)

def test_subset(k=10):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    subset = np.ones(n, np.bool)
    subset[-10:] = 0
    FS = forward_stepwise(X, Y, subset=subset,
                          covariance=0.5**2 * np.identity(n))

    for i in range(k):
        FS.next()

    print 'first %s variables selected' % k, FS.variables

    print 'pivots for last variable of 3rd selected model knowing that we performed %d steps of forward stepwise' % k

    print FS.model_pivots(3, saturated=True)
    print FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=5000, ndraw=5000)

    FS = forward_stepwise(X, Y, subset=subset)

    for i in range(k):
        FS.next()
    print FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=5000, ndraw=5000)

def test_BIC(k=10, do_sample=True):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = info_crit_stop(X, Y, 0.5, cost=np.log(n))
    final_model = len(FS.variables) - 1

    if do_sample:
        return [p[-1] for p in FS.model_pivots(final_model, saturated=False, burnin=5000, ndraw=5000)]
    else:
        saturated_pivots = FS.model_pivots(final_model)
        return [p[-1] for p in saturated_pivots]

def test_sequential(k=10):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    print sequential(X, Y, sigma=0.5, saturated=True)[1]
    print sequential(X, Y, sigma=0.5, saturated=False, ndraw=5000, burnin=5000)[1]
    print sequential(X, Y, saturated=False, ndraw=5000, burnin=5000)[1]
    
    # now use a subset of cases

    subset = np.ones(n, np.bool)
    subset[-10:] = 0
    print sequential(X, Y, sigma=0.5, saturated=False, ndraw=5000, burnin=5000,
                     subset=subset)[1]
    print sequential(X, Y, saturated=False, ndraw=5000, burnin=5000, subset=subset)[1]
    
def simulate_null():

    n, p = 100, 40
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_stepwise(X, Y, covariance=0.5**2 * np.identity(n))
    
    for i in range(5):
        FS.next()

    return [p[-1] for p in FS.model_pivots(3)]

def test_ecdf(nsim=1000, BIC=False):
    
    P = []
    for _ in range(nsim):
        if not BIC:
            P.extend(simulate_null())
        else:
            P.extend(test_BIC(do_sample=True))
    P = np.array(P)

    ecdf = sm.distributions.ECDF(P)

    plt.clf()
    plt.plot(ecdf.x, ecdf.y, linewidth=4, color='black')
    plt.show()
