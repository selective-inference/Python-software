import nose.tools as nt
import numpy as np

from selection.truncated.gaussian import truncated_gaussian, truncated_gaussian_old
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_for_test


intervals = [(-np.inf,-4.),(3.,np.inf)]

tg = truncated_gaussian(intervals)

X = np.linspace(-5,5,101)
F = [tg.cdf(x) for x in X]

def test_sigma():
    tg2 = truncated_gaussian_old(intervals, scale=2.)
    tg1 = truncated_gaussian_old(np.array(intervals)/2., scale=1.)

    Z = 3.5
    nt.assert_equal(np.around(float(tg1.cdf(Z/2.)), 3),
                    np.around(float(tg2.cdf(Z)), 3))
    np.testing.assert_equal(np.around(np.array(2 * tg1.equal_tailed_interval(Z/2,0.05)), 4),
                            np.around(np.array(tg2.equal_tailed_interval(Z,0.05)), 4))

@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_equal_tailed_coverage(burnin=None, 
                               ndraw=None,
                               nsim=1000):

    alpha = 0.25
    tg = truncated_gaussian_old([(2.3,np.inf)], scale=2)
    coverage = 0
    for i in range(nsim):
        while True:
            Z = np.random.standard_normal() * 2
            if Z > 2.3:
                break
        L, U = tg.equal_tailed_interval(Z, alpha)
        coverage += (U > 0) * (L < 0)
    SE = np.sqrt(alpha*(1-alpha)*nsim)
    print coverage
    nt.assert_true(np.fabs(coverage - (1-alpha)*nsim) < 2*SE)

@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_UMAU_coverage(burnin=None, 
                       ndraw=None,
                       nsim=1000):

    alpha = 0.25
    tg = truncated_gaussian_old([(2.3,np.inf)], scale=2)
    coverage = 0
    for i in range(nsim):
        while True:
            Z = np.random.standard_normal()*2
            if Z > 2.3:
                break
        L, U = tg.UMAU_interval(Z, alpha)
        coverage += (U > 0) * (L < 0)
    SE = np.sqrt(alpha*(1-alpha)*nsim)
    print coverage
    nt.assert_true(np.fabs(coverage - (1-alpha)*nsim) < 2*SE)
