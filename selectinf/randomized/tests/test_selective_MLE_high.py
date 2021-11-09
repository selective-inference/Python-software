import numpy as np
import nose.tools as nt

import regreg.api as rr


from ..lasso import (lasso,
                     split_lasso)

from ...base import (full_targets,
                     selected_targets,
                     debiased_targets)
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance,
                               cox_instance)


def test_full_targets(n=200,
                      p=1000,
                      signal_fac=0.5,
                      s=5,
                      sigma=3,
                      rho=0.4,
                      randomizer_scale=0.7,
                      full_dispersion=False):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    while True:
        signal = np.sqrt(signal_fac * 2 * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=True,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = np.linalg.norm(Y - X[:,nonzero].dot(np.linalg.pinv(X[:,nonzero]).dot(Y))) ** 2 / (n - nonzero.sum())

            if n > p:
                target_spec = full_targets(conv.loglike,
                                           conv.observed_soln,
                                           nonzero,
                                           dispersion=dispersion)
            else:
                target_spec = debiased_targets(conv.loglike,
                                               conv.observed_soln,
                                               nonzero,
                                               penalty=conv.penalty,
                                               dispersion=dispersion)

            conv.setup_inference(dispersion=dispersion)

            result = conv.selective_MLE(target_spec)[0]

            pval = result['pvalue']
            estimate = result['MLE']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
            print("estimate, intervals", estimate, intervals)

            coverage = (beta[nonzero] > intervals[:, 0]) * (beta[nonzero] < intervals[:, 1])
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals


def test_selected_targets(n=2000,
                          p=200,
                          signal_fac=1.2,
                          s=5,
                          sigma=2,
                          rho=0.7,
                          randomizer_scale=1.,
                          full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=True,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = 0.8 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())


        if nonzero.sum() > 0:

            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = np.linalg.norm(Y - X[:,nonzero].dot(np.linalg.pinv(X[:,nonzero]).dot(Y))) ** 2 / (n - nonzero.sum())

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result = conv.selective_MLE(target_spec)[0]

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals


def test_instance():
    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())

    L.setup_inference(dispersion=dispersion)

    target_spec = selected_targets(L.loglike,
                                   L.observed_soln,
                                   features=M,
                                   dispersion=dispersion)

    print("check shapes", target_spec.observed_target.shape, E.sum())

    result = L.selective_MLE(target_spec)[0]

    intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

    beta_target = np.linalg.pinv(X[:, M]).dot(X.dot(beta))

    coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

    return coverage

def test_selected_targets_disperse(n=500,
                                   p=100,
                                   s=5,
                                   sigma=1.,
                                   rho=0.4,
                                   randomizer_scale=1,
                                   full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = 1.

    while True:
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
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = np.linalg.norm(Y - X[:,nonzero].dot(np.linalg.pinv(X[:,nonzero]).dot(Y))) ** 2 / (n - nonzero.sum())

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result = conv.selective_MLE(target_spec)[0]

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals


def test_logistic(n=2000, 
                  p=200, 
                  signal_fac=10.,
                  s=5, 
                  rho=0.4, 
                  randomizer_scale=1):
    """
    Run approx MLE with selected targets on binomial data
    """

    inst, const = logistic_instance, lasso.logistic
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            conv.setup_inference(dispersion=1)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=1)

            result = conv.selective_MLE(target_spec)[0]
            estimate = result['MLE']
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])
            
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], intervals

def test_logistic_split(n=2000, 
                        p=200, 
                        signal_fac=10.,
                        s=5, 
                        rho=0.4, 
                        randomizer_scale=1):
    """
    Run approx MLE with selected targets on binomial data with data splitting
    """

    inst, const = logistic_instance, split_lasso.logistic
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     proportion=0.7)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=1)

            conv.setup_inference(dispersion=None)

            result = conv.selective_MLE(target_spec)[0]
            estimate = result['MLE']
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])
            
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], intervals

def test_poisson(n=2000, 
                 p=200, 
                 signal_fac=10.,
                 s=5, 
                 rho=0.4, 
                 randomizer_scale=1):
    """
    Run approx MLE with selected targets on Poisson data 
    """

    inst, const = poisson_instance, lasso.poisson
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=1)

            conv.setup_inference(dispersion=1)

            result = conv.selective_MLE(target_spec)[0]
            estimate = result['MLE']
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])
            
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], intervals

def test_poisson_split(n=2000, 
                       p=200, 
                       signal_fac=10.,
                       s=5, 
                       rho=0.4, 
                       randomizer_scale=1):
    """
    Run approx MLE with selected targets on Poisson data with data splitting
    """

    inst, const = poisson_instance, split_lasso.poisson
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     proportion=0.7)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=1)

            conv.setup_inference(dispersion=1)

            result = conv.selective_MLE(target_spec)[0]
            estimate = result['MLE']
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])
            
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], intervals

def test_cox(n=2000, 
             p=200, 
             signal_fac=10.,
             s=5, 
             rho=0.4, 
             randomizer_scale=1):
    """
    Run approx MLE with selected targets on survival data 
    """

    inst, const = cox_instance, lasso.coxph
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, T, S, beta = inst(n=n,
                             p=p,
                             signal=signal,
                             s=s,
                             equicorrelated=False,
                             rho=rho,
                             random_signs=True)[:4]

        n, p = X.shape

        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) 

        conv = const(X,
                     T,
                     S,
                     W,
                     randomizer_scale=randomizer_scale)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            cox_full = rr.glm.cox(X, T, S)
            full_hess = cox_full.hessian(conv.observed_soln)

            conv.setup_inference(dispersion=1)

            target_spec = selected_targets(conv.loglike, 
                                           conv.observed_soln,
                                           dispersion=1)

            result = conv.selective_MLE(target_spec)[0]
            estimate = result['MLE']
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])
            
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], intervals

def test_cox_split(n=2000, 
                   p=200, 
                   signal_fac=10.,
                   s=5, 
                   rho=0.4, 
                   randomizer_scale=1):
    """
    Run approx MLE with selected targets on survival data with data splitting
    """

    inst, const = cox_instance, split_lasso.coxph
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, T, S, beta = inst(n=n,
                             p=p,
                             signal=signal,
                             s=s,
                             equicorrelated=False,
                             rho=rho,
                             random_signs=True)[:4]

        n, p = X.shape

        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p))

        conv = const(X,
                     T,
                     S,
                     W,
                     proportion=0.7)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            cox_full = rr.glm.cox(X, T, S)
            full_hess = cox_full.hessian(conv.observed_soln)

            conv.setup_inference(dispersion=1)

            target_spec = selected_targets(conv.loglike, 
                                           conv.observed_soln,
                                           dispersion=1)

            result = conv.selective_MLE(target_spec)[0]
            estimate = result['MLE']
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])
            
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], intervals
        
def test_scale_invariant_split(n=200, 
                               p=20, 
                               signal_fac=10.,
                               s=5, 
                               sigma=3, 
                               rho=0.4, 
                               randomizer_scale=1,
                               full_dispersion=True,
                               seed=2):
    """
    Confirm Gaussian version is appropriately scale invariant with data splitting
    """

    inst, const = gaussian_instance, split_lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    results = []

    scales = [1, 5]
    for scale in scales:

        np.random.seed(seed)
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        Y *= scale; beta *= scale
        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_
        print('W', W[0]/scale)
        conv = const(X,
                     Y,
                     W,
                     proportion=0.7)

        signs = conv.fit()
        nonzero = signs != 0
        print('nonzero', np.where(nonzero)[0])
        print('feature_weights', conv.feature_weights[0] / scale)
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        #print('dispersion', target_spec.dispersion/scale**2)
        print('target', target_spec.observed_target[0]/scale)
        print('cov_target', target_spec.cov_target[0,0]/scale**2)
        print('regress_target_score',  target_spec.regress_target_score[0,0]/scale**2)


        result = conv.selective_MLE(target_spec)[0]

        print(result['MLE'] / scale)
        results.append(result)

    assert np.allclose(results[0]['MLE'] / scales[0],
                       results[1]['MLE'] / scales[1])
    assert np.allclose(results[0]['SE'] / scales[0],
                       results[1]['SE'] / scales[1])
    assert np.allclose(results[0]['upper_confidence'] / scales[0],
                       results[1]['upper_confidence'] / scales[1])
    assert np.allclose(results[0]['lower_confidence'] / scales[0],
                       results[1]['lower_confidence'] / scales[1])
    assert np.allclose(results[0]['Zvalue'],
                       results[1]['Zvalue'])
    assert np.allclose(results[0]['pvalue'],
                       results[1]['pvalue'])

def test_scale_invariant(n=200, 
                         p=20, 
                         signal_fac=10.,
                         s=5, 
                         sigma=3, 
                         rho=0.4, 
                         randomizer_scale=1,
                         full_dispersion=True,
                         seed=2):
    """
    Confirm Gaussian version is appropriately scale invariant
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    results = []

    scales = [1, 5]
    for scale in scales:

        np.random.seed(seed)
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        Y *= scale; beta *= scale
        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_
        print('W', W[0]/scale)
        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print('nonzero', np.where(nonzero)[0])
        print('feature_weights', conv.feature_weights[0] / scale)
        print('perturb', conv._initial_omega[0] / scale)
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        #print('dispersion', target_spec.dispersion/scale**2)
        print('target', target_spec.observed_target[0]/scale)
        print('cov_target', target_spec.cov_target[0,0]/scale**2)
        print('regress_target_score',  target_spec.regress_target_score[0,0]/scale**2)
        
        result = conv.selective_MLE(target_spec)[0]

        print(result['MLE'] / scale)
        results.append(result)

    assert np.allclose(results[0]['MLE'] / scales[0],
                       results[1]['MLE'] / scales[1])
    assert np.allclose(results[0]['SE'] / scales[0],
                       results[1]['SE'] / scales[1])
    assert np.allclose(results[0]['upper_confidence'] / scales[0],
                       results[1]['upper_confidence'] / scales[1])
    assert np.allclose(results[0]['lower_confidence'] / scales[0],
                       results[1]['lower_confidence'] / scales[1])
    assert np.allclose(results[0]['Zvalue'],
                       results[1]['Zvalue'])
    assert np.allclose(results[0]['pvalue'],
                       results[1]['pvalue'])
    

