import numpy as np

from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.instance import gaussian_instance
from selection.algorithms.forward_step import forward_step, info_crit_stop, data_carving_IC
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_FS(k=10, ndraw=5000, burnin=5000):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_step(X, Y, covariance=0.5**2 * np.identity(n))

    for i in range(k):
        FS.next(compute_pval=True)

    print('first %s variables selected' % k, FS.variables)

    print('pivots for 3rd selected model knowing that we performed %d steps of forward stepwise' % k)

    print(FS.model_pivots(3))
    print(FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=burnin, ndraw=ndraw))
    print(FS.model_quadratic(3))

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_FS_unknown(k=10, ndraw=5000, burnin=5000):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_step(X, Y)

    for i in range(k):
        FS.next()

    print('first %s variables selected' % k, FS.variables)

    print('pivots for last variable of 3rd selected model knowing that we performed %d steps of forward stepwise' % k)

    print(FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=burnin, ndraw=ndraw))

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_subset(k=10, ndraw=5000, burnin=5000):

    n, p = 100, 200
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    subset = np.ones(n, np.bool)
    subset[-10:] = 0
    FS = forward_step(X, Y, subset=subset,
                          covariance=0.5**2 * np.identity(n))

    for i in range(k):
        FS.next()

    print('first %s variables selected' % k, FS.variables)

    print('pivots for last variable of 3rd selected model knowing that we performed %d steps of forward stepwise' % k)

    print(FS.model_pivots(3, saturated=True))
    print(FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=burnin, ndraw=ndraw))

    FS = forward_step(X, Y, subset=subset)

    for i in range(k):
        FS.next()
    print(FS.model_pivots(3, saturated=False, which_var=[FS.variables[2]], burnin=burnin, ndraw=ndraw))

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_BIC(do_sample=True, ndraw=8000, burnin=2000, 
             force=False):

    X, Y, beta, active, sigma = gaussian_instance()
    n, p = X.shape
    FS = info_crit_stop(Y, X, sigma, cost=np.log(n))
    final_model = len(FS.variables) 

    active = set(list(active))
    if active.issubset(FS.variables) or force:
        which_var = [v for v in FS.variables if v not in active]

        if do_sample:
            return [pval[-1] for pval in FS.model_pivots(final_model, saturated=False, burnin=burnin, ndraw=ndraw, which_var=which_var)]
        else:
            saturated_pivots = FS.model_pivots(final_model, which_var=which_var)
            return [pval[-1] for pval in saturated_pivots]
    return []

def simulate_null(saturated=True, ndraw=8000, burnin=2000):

    n, p = 100, 40
    X = np.random.standard_normal((n,p)) + 0.4 * np.random.standard_normal(n)[:,None]
    X /= (X.std(0)[None,:] * np.sqrt(n))
    
    Y = np.random.standard_normal(100) * 0.5
    
    FS = forward_step(X, Y, covariance=0.5**2 * np.identity(n))
    
    for i in range(5):
        FS.next()

    return [p[-1] for p in FS.model_pivots(3, saturated=saturated, ndraw=ndraw, burnin=burnin)]

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10, nsim=10)
def test_ecdf(nsim=1000, BIC=False,
              saturated=True,
              burnin=2000,
              ndraw=8000):
    
    P = []
    for _ in range(nsim):
        if not BIC:
            P.extend(simulate_null(saturated=saturated, ndraw=ndraw, burnin=burnin))
        else:
            P.extend(test_BIC(do_sample=True, ndraw=ndraw, burnin=burnin))
    P = np.array(P)

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10, nsim=10)
def test_data_carving_IC(nsim=500,
                         n=100,
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

    while counter < nsim:
        counter += 1
        X, y, beta, active, sigma = gaussian_instance(n=n, 
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

            carve_split = [(r[1], r[3]) for r in results]
            carve = np.array(carve_split)[:,0]
            split = np.array(carve_split)[:,1]

            Xa = X[:,FS.variables[:-1]]
            truth = np.dot(np.linalg.pinv(Xa), mu) 

            split_coverage = []
            carve_coverage = []
            for result, t in zip(results, truth):
                _, _, ci, _, si = result
                carve_coverage.append((ci[0] < t) * (t < ci[1]))
                split_coverage.append((si[0] < t) * (t < si[1]))

            print(carve, 'carve')
            print(split, 'split')
            print(results, 'results')
            return ([carve[j] for j, i in enumerate(FS.active) if i >= s], 
                    [split[j] for j, i in enumerate(FS.active) if i >= s], 
                    [carve[j] for j, i in enumerate(FS.active) if i < s], 
                    [split[j] for j, i in enumerate(FS.active) if i < s], 
                    counter, carve_coverage, split_coverage)


@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_full_pvals(n=100, p=40, rho=0.3, snr=4, ndraw=8000, burnin=2000):

    X, y, beta, active, sigma = gaussian_instance(n=n, p=p, snr=snr, rho=rho)
    FS = forward_step(X, y, covariance=sigma**2 * np.identity(n))

    from scipy.stats import norm as ndist
    pval = []
    completed_yet = False
    for i in range(min(n, p)):
        FS.next()
        var_select, pval_select = FS.model_pivots(i+1, alternative='twosided',
                                                  which_var=[FS.variables[-1]],
                                                  saturated=False,
                                                  burnin=burnin,
                                                  ndraw=ndraw)[0]
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

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_mcmc_tests(n=100, p=40, s=4, rho=0.3, snr=5, ndraw=None, burnin=2000,
                    nstep=200,
                    method='serial'):

    X, y, beta, active, sigma = gaussian_instance(n=n, p=p, snr=snr, rho=rho, s=s)
    FS = forward_step(X, y, covariance=sigma**2 * np.identity(n))

    extra_steps = 4

    null_rank, alt_rank = None, None

    for i in range(min(n, p)):
        FS.next()

        if extra_steps <= 0:
            null_rank = FS.mcmc_test(i+1, variable=FS.variables[i-2], 
                                     nstep=nstep,
                                     burnin=burnin,
                                     method="serial")
            alt_rank = FS.mcmc_test(i+1, variable=FS.variables[0], 
                                    burnin=burnin,
                                    nstep=nstep, 
                                    method="parallel")
            break

        if set(active).issubset(FS.variables):
            extra_steps -= 1

    return null_rank, alt_rank

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_independence_null_mcmc(n=100, p=40, s=4, rho=0.5, snr=5, 
                                ndraw=None, burnin=2000,
                                nstep=200,
                                method='serial'):

    X, y, beta, active, sigma = gaussian_instance(n=n, p=p, snr=snr, rho=rho, s=s)
    FS = forward_step(X, y, covariance=sigma**2 * np.identity(n))

    extra_steps = 4
    completed = False

    null_ranks = []
    for i in range(min(n, p)):
        FS.next()

        if completed and extra_steps > 0:
            null_rank = FS.mcmc_test(i+1, variable=FS.variables[-1], 
                                     nstep=nstep,
                                     burnin=burnin,
                                     method="serial")
            null_ranks.append(int(null_rank))

        if extra_steps <= 0:
            break

        if set(active).issubset(FS.variables):
            extra_steps -= 1
            completed = True

    return tuple(null_ranks)

