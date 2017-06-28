import itertools

import numpy as np
import numpy.testing.decorators as dec

from selection.tests.instance import gaussian_instance
from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.algorithms.lasso import lasso
from selection.algorithms.covtest import covtest, selected_covtest
from selection.constraints.affine import gibbs_test
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_covtest(nsim=None, ndraw=8000, burnin=2000):

    n, p = 30, 50
    X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,None]
    X /= X.std(0)[None,:]
    Y = np.random.standard_normal(n) * 1.5 

    for exact, covariance in itertools.product([True, False],
                                               [None, np.identity(n)]):
        con, pval, idx, sign = covtest(X, Y, sigma=1.5, exact=exact,
                                       covariance=covariance)
    for covariance in [None, np.identity(n)]:
        con, pval, idx, sign = selected_covtest(X, Y, sigma=1.5,
                                                covariance=covariance,
                                                ndraw=ndraw,
                                                burnin=burnin)

    con, pval, idx, sign = selected_covtest(X, Y)

    return pval

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsim=5, ndraw=10, burnin=20)
def test_tilting(nsim=100, ndraw=50000, burnin=10000):

    P = []
    covered0 = 0
    coveredA = 0
    screen = 0

    for i in range(nsim):
        X, Y, beta, active, sigma = gaussian_instance(n=20, p=30)

        Y0 = np.random.standard_normal(X.shape[0]) * sigma

        # null pvalues and intervals

        cone, pvalue, idx, sign = selected_covtest(X, Y0, sigma=sigma)
        eta = X[:,idx] * sign
        p1, _, _, fam = gibbs_test(cone, Y0, eta, 
                                   ndraw=ndraw,
                                   burnin=burnin,
                                   alternative='twosided',
                                   sigma_known=True,
                                   tilt=eta,
                                   UMPU=False)

        observed_value = (Y0 * eta).sum()
        lower_lim, upper_lim = fam.equal_tailed_interval(observed_value)
        lower_lim_final = np.dot(eta, np.dot(cone.covariance, eta)) * lower_lim
        upper_lim_final = np.dot(eta, np.dot(cone.covariance, eta)) * upper_lim
        covered0 += (lower_lim_final < 0) * (upper_lim_final > 0)
        print(covered0 / (i + 1.), 'coverage0')

        # compare to no tilting

        p2 = gibbs_test(cone, Y0, X[:,idx] * sign,
                        ndraw=ndraw,
                        burnin=burnin,
                        alternative='twosided',
                        sigma_known=True,
                        tilt=None,
                        UMPU=False)[0]
        print(p2, 'huh')
        P.append((p1, p2))
        Pa = np.array(P)

        # p1 and p2 should be very close, so have high correlation
        print(np.corrcoef(Pa.T)[0,1], 'correlation')

        # they should also look uniform -- mean should be about 0.5, sd about 0.29

        print(np.mean(Pa, 0), 'mean of nulls')
        print(np.std(Pa, 0), 'sd of nulls')

        # alternative intervals

        mu = 3 * X[:,0] * sigma
        YA = np.random.standard_normal(X.shape[0]) * sigma + mu 

        cone, pvalue, idx, sign = selected_covtest(X, YA, sigma=sigma)
        _, _, _, fam = gibbs_test(cone, YA, X[:,idx] * sign,
                                  ndraw=ndraw,
                                  burnin=burnin,
                                  alternative='greater',
                                  sigma_known=True,
                                  tilt=eta)

        if idx == 0:
            screen += 1

            eta = X[:,0] * sign
            observed_value = (YA * eta).sum()
            target = (eta * mu).sum()
            lower_lim, upper_lim = fam.equal_tailed_interval(observed_value)
            lower_lim_final = np.dot(eta, np.dot(cone.covariance, eta)) * lower_lim
            upper_lim_final = np.dot(eta, np.dot(cone.covariance, eta)) * upper_lim
            print(lower_lim_final, upper_lim_final, target)
            coveredA += (lower_lim_final < target) * (upper_lim_final > target)
            print(coveredA / (screen * 1.), 'coverageA')

        print(screen / (i + 1.), 'screening')

