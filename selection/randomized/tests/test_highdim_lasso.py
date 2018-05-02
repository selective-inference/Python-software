from __future__ import division, print_function

import numpy as np
import nose.tools as nt

import regreg.api as rr

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(selectiveInference)')

from ..lasso import lasso
from ...tests.instance import gaussian_instance
from ...algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
import matplotlib.pyplot as plt

def test_highdim_lasso(n=500, p=200, signal_fac=1.5, s=5, sigma=3, target='full', rho=0.4, randomizer_scale=1, ndraw=5000, burnin=1000):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)
    W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma_

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer_scale=randomizer_scale * sigma_)
    
    signs = conv.fit()
    nonzero = signs != 0

    _, pval, intervals = conv.summary(target=target,
                                      ndraw=ndraw,
                                      burnin=burnin, 
                                      compute_intervals=False)
        
    return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0]

def test_all_targets(n=100, p=20, signal_fac=1.5, s=5, sigma=3, rho=0.4):
    for target in ['full', 'selected', 'debiased']:
        test_highdim_lasso(n=n, p=p, signal_fac=signal_fac, s=s, sigma=sigma, rho=rho)

def test_sqrt_highdim_lasso(n=500, p=200, signal_fac=1.5, s=5, sigma=3, full=True, rho=0.4, randomizer_scale=1., ndraw=5000, burnin=1000, 
                            ridge_term=None, compare_to_lasso=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.sqrt_lasso
    signal = np.sqrt(signal_fac * 2 * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p, 
                      signal=signal, 
                      s=s, 
                      equicorrelated=False, 
                      rho=rho, 
                      sigma=sigma, 
                      random_signs=True)[:3]

    if ridge_term is None:
        mean_diag = np.mean((X**2).sum(0))
        ridge_term = (np.sqrt(mean_diag) / np.sqrt(n)) * np.sqrt(n / (n - 1.))

    W = np.ones(X.shape[1]) * choose_lambda(X) * 0.7

    perturb = np.random.standard_normal(p) * randomizer_scale / np.sqrt(n)

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer_scale=randomizer_scale / np.sqrt(n),
                 perturb=perturb,
                 ridge_term=ridge_term)
    
    signs = conv.fit()
    nonzero = signs != 0

    # sanity check

    if compare_to_lasso:
        q_term = rr.identity_quadratic(ridge_term, 0, -perturb, 0)

        soln2, sqrt_loss = solve_sqrt_lasso(X, Y, W, solve_args={'min_its':1000}, quadratic=q_term, force_fat=True)
        soln = conv.initial_soln

        denom = np.linalg.norm(Y - X.dot(soln))
        new_weights = W * denom
        loss = rr.glm.gaussian(X, Y)
        pen = rr.weighted_l1norm(new_weights, lagrange=1.)
        prob = rr.simple_problem(loss, pen)

        rescaledQ = rr.identity_quadratic(ridge_term * denom,
                                          0,
                                          -perturb * denom,
                                          0)

        soln3 = prob.solve(quadratic=rescaledQ, min_its=1000, tol=1.e-12)
        np.testing.assert_allclose(conv._initial_omega, perturb * denom)
        np.testing.assert_allclose(soln, soln2)
        np.testing.assert_allclose(soln, soln3)

    if full:
        _, pval, intervals = conv.summary(target="full",
                                          ndraw=ndraw,
                                          burnin=burnin, 
                                          compute_intervals=False)
    else:
        _, pval, intervals = conv.summary(target="selected",
                                          ndraw=ndraw,
                                          burnin=burnin, 
                                          compute_intervals=False)

    return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0]

def test_compareR(n=200, p=10, signal=np.sqrt(4) * np.sqrt(2 * np.log(10)), s=5, ndraw=5000, burnin=1000, param=True, sigma=3):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    X, Y, beta = inst(n=n, p=p, signal=signal, s=s, equicorrelated=False, rho=0.2, sigma=sigma, random_signs=True)[:3]

    n, p = X.shape

    W = np.ones(X.shape[1]) * np.sqrt(1.5 * np.log(p)) * sigma
    randomizer_scale = np.std(Y) * .5 * np.sqrt(n / (n - 1.)) # to agree more exactly with R

    pval, vars, rand, active, soln, ridge_term, cond_cov, cond_mean = Rpval(X, Y, W, randomizer_scale)

    conv = const(X, 
                 Y, 
                 W, 
                 randomizer_scale=randomizer_scale)
    
    signs = conv.fit(perturb=rand, solve_args={'min_its':500, 'tol':1.e-12})

    assert np.fabs(conv.ridge_term - ridge_term) / ridge_term < 1.e-4

    assert np.fabs(soln - conv.initial_soln).max() / np.fabs(soln).max() < 1.e-3


    nonzero = signs != 0

    assert np.linalg.norm(conv.sampler.affine_con.covariance - cond_cov) / np.linalg.norm(cond_cov) < 1.e-3
    assert np.linalg.norm(conv.sampler.affine_con.mean - cond_mean[:,0]) / np.linalg.norm(cond_mean[:,0]) < 1.e-3


def main(nsim=500, n=500, p=200, sqrt=False, target='full', sigma=3):

    P0, PA = [], []
    from statsmodels.distributions import ECDF

    for i in range(nsim):
        if True: # try:
            if not sqrt:
                p0, pA = test_highdim_lasso(n=n, p=p, target=target, sigma=sigma)
            else:
                p0, pA = test_sqrt_highdim_lasso(n=n, p=p, target=target, compare_to_lasso=False)
        else: # except:
            p0, pA = [], []
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(P0), np.std(P0), np.mean(np.array(PA) < 0.05), 'null pvalue + power')
    
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
        rpy.r('soln = selectiveInference:::randomizedLasso(X, Y, lam, noise_scale=noise_scale, kkt_tol=1.e-8, parameter_tol=1.e-8)')
    else:
        rpy.r('soln = selectiveInference:::randomizedLasso(X, Y, lam)')
    rpy.r('targets=selectiveInference:::set.target(soln, type="full")')
    #rpy.r('rand_inf = selectiveInference:::randomizedLassoInf(soln, sampler="norejection", targets=targets, nsample=5000, burnin=1000)')
    rpy.r('rand_inf = selectiveInference:::randomizedLassoInf(soln, sampler="restrictedMVN", targets=targets, nsample=5000, burnin=2000)')

    pval = np.asarray(rpy.r('rand_inf$pvalues'))
    vars = np.asarray(rpy.r('soln$active_set')) - 1 
    cond_cov = np.asarray(rpy.r('soln$law$cond_cov'))
    cond_mean = np.asarray(rpy.r('soln$law$cond_mean'))
    rand = np.asarray(rpy.r('soln$perturb'))
    active =  np.asarray(rpy.r('soln$active')) - 1
    soln = np.asarray(rpy.r('soln$soln'))
    ridge = rpy.r('soln$ridge_term')

    numpy2ri.deactivate()
    return pval, vars, rand, active, soln, ridge, cond_cov, cond_mean


# if __name__ == "__main__":
#     main()
