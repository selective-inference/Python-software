import numpy as np
import numpy.testing.decorators as dec
from selection.constraints.affine import constraints
from selection.sampling.sequential import sample
from scipy.stats import norm as ndist
import statsmodels.api as sm

from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt

from selection.tests.decorators import set_sampling_params_iftrue, set_seed_for_test

@dec.slow
@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_sequentially_constrained(burnin=None, ndraw=1000, nsim=None):
    S = -np.identity(10)[:3]
    b = -6 * np.ones(3)
    C = constraints(S, b)
    W = sample(C, nsim, temps=np.linspace(0, 200, 1001))
    U = np.linspace(0, 1, 101)
    D = sm.distributions.ECDF((ndist.cdf(W[0]) - ndist.cdf(6)) / ndist.sf(6))
    plt.plot(U, D(U))
