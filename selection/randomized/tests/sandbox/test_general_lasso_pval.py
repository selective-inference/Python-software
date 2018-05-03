import numpy as np
import nose.tools as nt
import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(selectiveInference)')

from selection.randomized.lasso import lasso
from selection.tests.instance import gaussian_instance
import matplotlib.pyplot as plt

n, p = 500, 20

def test_condition_subgrad(n=n, p=p, signal=np.sqrt(2 * np.log(p)), s=5, ndraw=5000, burnin=1000, param=True, sigma=1, full=True, rho=0.2, useR=True, randomizer_scale=1):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    n, p = X.shape

    W = np.ones(X.shape[1]) * 1.5 * sigma

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer='gaussian', 
                 parametric_cov_estimator=param,
                 randomizer_scale=randomizer_scale)
    
    nboot = 2000
    signs = conv.fit(nboot=nboot)
    nonzero = signs != 0
    conv.decompose_subgradient(condition=np.ones(p, np.bool))

    if full:
        selected = np.ones(p, np.bool)
        keep = nonzero
    else:
        selected = nonzero
        selected_idx = np.nonzero(selected)[0]
        keep = np.ones(selected_idx.shape[0], np.bool)

    _, pval, intervals = conv.summary(selected,
                                      ndraw=ndraw,
                                      burnin=burnin, compute_intervals=False,
                                      subset=keep)

    if full:
        if not useR:
            return pval[beta[keep] == 0], pval[beta[keep] != 0]
        else:
            pval, selected_idx = Rpval(X, Y, W, randomizer_scale)[:2]
            return [p for j, p in zip(selected_idx, pval) if beta[j] == 0], [p for j, p in zip(selected_idx, pval) if beta[j] != 0]
    else:
        return [p for j, p in zip(selected_idx, pval) if beta[j] == 0], [p for j, p in zip(selected_idx, pval) if beta[j] != 0]

def test_compareR(n=n, p=p, signal=np.sqrt(4) * np.sqrt(2 * np.log(p)), s=5, ndraw=5000, burnin=1000, param=True, sigma=3):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    X, Y, beta = inst(n=n, p=p, signal=signal, s=s, equicorrelated=False, rho=0.2, sigma=sigma, random_signs=True)[:3]

    n, p = X.shape

    W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma
    randomizer_scale = np.std(Y) * .5

    L, O, rand, active, soln, ridge_term, cond_cov, cond_mean = Rpval(X, Y, W, randomizer_scale)[2:]
    implied_prec = L.T.dot(L) / randomizer_scale**2

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer='gaussian', 
                 parametric_cov_estimator=param,
                 randomizer_scale=randomizer_scale)
    
    nboot = 2000

    signs = conv.fit(nboot=nboot, perturb=rand, solve_args={'min_its':500})

    assert np.fabs(conv._view.epsilon - np.sqrt((n - 1.) / n) * ridge_term) / ridge_term < 1.e-4

    assert np.fabs(soln - conv._view.initial_soln).max() / np.fabs(soln).max() < 1.e-3


    nonzero = signs != 0
    print(nonzero.sum())

    print(np.diag(np.linalg.inv(X.T.dot(X)) * sigma**2))
    
    conv.decompose_subgradient(condition=np.ones(p, np.bool))

    assert np.linalg.norm(np.linalg.inv(conv._view.sampler.affine_con.covariance) - implied_prec) / np.linalg.norm(implied_prec) < 1.e-3

    assert np.linalg.norm(conv._view.sampler.affine_con.mean - cond_mean[:,0]) / np.linalg.norm(cond_mean[:,0]) < 1.e-3
    assert np.linalg.norm(conv._view.sampler.affine_con.covariance - cond_cov) / np.linalg.norm(cond_cov) < 1.e-3

    full = False

    if full:
        selected = np.ones(p, np.bool)
        keep = nonzero
    else:
        selected = nonzero
        selected_idx = np.nonzero(selected)[0]
        keep = True

    _, pval, intervals = conv.summary(selected,
                                      ndraw=ndraw,
                                      burnin=burnin, compute_intervals=False)

    pval = np.asarray(pval)
    pval = 2 * np.minimum(pval, 1 - pval)

#    if not full:
#        pval, selected_idx = Rpval(X, Y, W, randomizer_scale)[:2]

    if full:
        return pval[nonzero][beta[nonzero] == 0], pval[nonzero][beta[nonzero] != 0]
#        return pval[nonzero][beta[nonzero] == 0], pval[nonzero][beta[nonzero] != 0]
    else:
        return [p for j, p in zip(selected_idx, pval) if beta[j] == 0], [p for j, p in zip(selected_idx, pval) if beta[j] != 0]

def main(nsim=500):

    P0, PA = [], []
    from statsmodels.distributions import ECDF

    for i in range(nsim):
        try:
            p0, pA = test_condition_subgrad(n=200, p=10)
        except:
            p0, pA = [], []
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(P0), np.std(P0), np.mean(np.array(PA) < 0.05))
    
        if i % 3 == 0 and i > 0:
            U = np.linspace(0, 1, 101)
            plt.clf()
            if len(P0) > 0:
                plt.plot(U, ECDF(P0)(U))
            if len(PA) > 0:
                plt.plot(U, ECDF(PA)(U), 'r')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.savefig("plot.pdf")
    plt.show()

def Rpval(X, Y, W, noise_scale=None):
    numpy2ri.activate()
    rpy.r.assign('X', X)
    rpy.r.assign('Y', Y)
    rpy.r.assign('lam', W)
    if noise_scale is not None:
        rpy.r.assign('noise_scale', noise_scale)
        rpy.r('soln = selectiveInference:::randomizedLasso(X, Y, lam, noise_scale=noise_scale)')
    else:
        rpy.r('soln = selectiveInference:::randomizedLasso(X, Y, lam)')
    rpy.r('full_targets=selectiveInference:::set.target(soln,type="full")')
    print('here')
    rpy.r('rand_inf = selectiveInference:::randomizedLassoInf(soln, sampler="restrictedMVN", full_targets=full_targets, nsample=10000, burnin=3000)')
    pval = np.asarray(rpy.r('rand_inf$pvalues'))
    vars = np.asarray(rpy.r('soln$active_set')) - 1 

    L = np.asarray(rpy.r('soln$law$sampling_transform$linear_term'))
    O = np.asarray(rpy.r('soln$law$sampling_transform$offset_term'))
    cond_cov = np.asarray(rpy.r('soln$law$cond_cov'))
    cond_mean = np.asarray(rpy.r('soln$law$cond_mean'))
    rand = np.asarray(rpy.r('soln$perturb'))
    active =  np.asarray(rpy.r('soln$active')) - 1
    soln = np.asarray(rpy.r('soln$soln'))
    rpy.r('print(names(soln))')
    rpy.r('print(names(soln$law))')
    ridge = rpy.r('soln$ridge_term')

    try:
        #pval = 2 * np.minimum(pval, 1 - pval)
        return pval, vars, L, O, rand, active, soln, ridge, cond_cov, cond_mean
    except:
        return [], []


# if __name__ == "__main__":
#     main()
