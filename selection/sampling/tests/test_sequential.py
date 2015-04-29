import numpy as np
import numpy.testing.decorators as dec
from selection.constraints.affine import constraints
import statsmodels.api as sm

#@dec.slow()
def test_sequentially_constrained():
    S = -np.identity(10)[:3]
    b = -6 * np.ones(3)
    C = constraints(S, b)
    W = sample(C, 5000, temps=np.linspace(0, 200, 1001))
    U = np.linspace(0, 1, 101)
    D = sm.distributions.ECDF((ndist.cdf(W[0]) - ndist.cdf(6)) / ndist.sf(6))
    plt.plot(U, D(U))
