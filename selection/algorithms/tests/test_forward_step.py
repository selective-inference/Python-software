import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from selection.algorithms.lasso import instance
from selection.algorithms.forward_step import forward_stepwise, info_crit_stop, sequential, data_carving_IC

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
    
    FS = info_crit_stop(Y, X, 0.5, cost=np.log(n))
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
    
def simulate_null(saturated=True):

    n, p = 100, 40
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_stepwise(X, Y, covariance=0.5**2 * np.identity(n))
    
    for i in range(5):
        FS.next()

    return [p[-1] for p in FS.model_pivots(3, saturated=saturated,
                                           use_new=False)]

def test_ecdf(nsim=1000, BIC=False,
              saturated=True):
    
    P = []
    for _ in range(nsim):
        if not BIC:
            P.extend(simulate_null(saturated=saturated))
        else:
            P.extend(test_BIC(do_sample=True))
    P = np.array(P)

    ecdf = sm.distributions.ECDF(P)

    plt.clf()
    plt.plot(ecdf.x, ecdf.y, linewidth=4, color='black')
    plt.show()

def test_data_carving_IC(n=100,
                         p=200,
                         s=7,
                         sigma=5,
                         rho=0.3,
                         snr=7.,
                         split_frac=0.9,
                         ndraw=5000,
                         burnin=1000, 
                         df=np.inf,
                         coverage=0.90,
                         compute_intervals=False):

    counter = 0

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr, 
                                             df=df)
        mu = np.dot(X, beta)
        splitn = int(n*split_frac)
        indices = np.arange(n)
        np.random.shuffle(indices)
        stage_one = indices[:splitn]

        FS = info_crit_stop(y, X, sigma, cost=np.log(n), subset=stage_one)

        if set(range(s)).issubset(FS.active):
            results, FS = data_carving_IC(y, X, sigma,
                                          stage_one=stage_one,
                                          splitting=True, 
                                          ndraw=ndraw,
                                          burnin=burnin,
                                          coverage=coverage,
                                          compute_intervals=compute_intervals,
                                          cost=np.log(n))

            carve = [r[1] for r in results]
            split = [r[3] for r in results]

            Xa = X[:,FS.variables[:-1]]
            truth = np.dot(np.linalg.pinv(Xa), mu) 

            split_coverage = []
            carve_coverage = []
            for result, t in zip(results, truth):
                _, _, ci, _, si = result
                carve_coverage.append((ci[0] < t) * (t < ci[1]))
                split_coverage.append((si[0] < t) * (t < si[1]))

            return ([carve[j] for j, i in enumerate(FS.active) if i >= s], 
                    [split[j] for j, i in enumerate(FS.active) if i >= s], 
                    [carve[j] for j, i in enumerate(FS.active) if i < s], 
                    [split[j] for j, i in enumerate(FS.active) if i < s], 
                    counter, carve_coverage, split_coverage)


def test_full_pvals(n=100, p=40, rho=0.3, snr=4):

    X, y, beta, active, sigma = instance(n=n, p=p, snr=snr, rho=rho)
    FS = forward_stepwise(X, y, covariance=sigma**2 * np.identity(n))

    from scipy.stats import norm as ndist
    pval = []
    completed_yet = False
    for i in range(min(n, p)):
        FS.next()
        var_select, pval_select = FS.model_pivots(i+1, alternative='twosided',
                                                  which_var=[FS.variables[-1]],
                                                  saturated=False,
                                                  burnin=2000,
                                                  ndraw=8000)[0]
        pval_saturated = FS.model_pivots(i+1, alternative='twosided',
                                         which_var=[FS.variables[-1]],
                                         saturated=True)[0][1]

        # now, nominal ones

        LSfunc = np.linalg.pinv(FS.X[:,FS.variables])
        Z = np.dot(LSfunc[-1], FS.Y) / (np.linalg.norm(LSfunc[-1]) * sigma)
        pval_nominal = 2 * ndist.sf(np.fabs(Z))
        pval.append((var_select, pval_select, pval_saturated, pval_nominal))
            
        if set(active).issubset(np.array(pval)[:,0]) and not completed_yet:
            completed_yet = True
            completion_index = i + 1

    return X, y, beta, active, sigma, np.array(pval), completion_index
