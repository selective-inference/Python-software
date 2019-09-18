import numpy as np
import numpy.testing.decorators as dec
from scipy.stats import norm as ndist

from ...constraints.affine import constraints
from ..sequential import sample
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from ...tests.flags import SMALL_SAMPLES, SET_SEED

@dec.slow
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, nsim=10)
def test_sequentially_constrained(ndraw=100, nsim=50):
    S = -np.identity(10)[:3]
    b = -6 * np.ones(3)
    C = constraints(S, b)
    W = sample(C, nsim, temps=np.linspace(0, 200, 1001))
    U = np.linspace(0, 1, 101)

