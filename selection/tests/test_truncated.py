import nose.tools as nt
import numpy as np
from selection.truncated import truncated_gaussian

intervals = [(-np.inf,-4.),(3.,np.inf)]

tg = truncated_gaussian(intervals)

X = np.linspace(-5,5,101)
F = [tg.cdf(x) for x in X]

def test_sigma():
    tg2 = truncated_gaussian(intervals, sigma=2.)
    tg1 = truncated_gaussian(np.array(intervals)/2., sigma=1.)

    Z = 3.5
    nt.assert_equal(np.around(float(tg1.cdf(Z/2.)), 3),
                    np.around(float(tg2.cdf(Z)), 3))
    np.testing.assert_equal(np.around(np.array(2 * tg1.equal_tailed_interval(Z/2,0.05)), 4),
                            np.around(np.array(tg2.equal_tailed_interval(Z,0.05)), 4))

def test_equal_tailed_coverage():

    alpha = 0.15
    nsim = 1000
    tg = truncated_gaussian([(2.3,np.inf)], sigma=2)
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

def test_UMAU_coverage():

    alpha = 0.15
    nsim = 1000
    tg = truncated_gaussian([(2.3,np.inf)], sigma=2)
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
