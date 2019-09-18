import numpy as np
from scipy.stats import f as fdist, t as tdist

from ..F import sf_F
from ..T import sf_T

def test_F():

    f1 = sf_F(3.,20.,1)
    f2 = fdist(3.,20.)

    V = np.linspace(1,7,201)
    V1 = [float(f1(v)) for v in V]
    V2 = f2.sf(V)
    np.testing.assert_allclose(V1, V2)

    V = np.linspace(1,7,11)
    V1 = [float(f1(u,v)) for u,v in zip(V[:-1],V[1:])]
    V2 = [f2.sf(u)-f2.sf(v) for u,v in zip(V[:-1],V[1:])]
    np.testing.assert_allclose(V1, V2)

def test_T():

    f1 = sf_T(20.)
    f2 = tdist(20.)

    V = np.linspace(-2,3,201)
    V1 = [float(f1(v)) for v in V]
    V2 = f2.sf(V)
    np.testing.assert_allclose(V1, V2)

    V = np.linspace(-2,3,11)
    V1 = [float(f1(u,v)) for u,v in zip(V[:-1],V[1:])]
    V2 = [f2.sf(u)-f2.sf(v) for u,v in zip(V[:-1],V[1:])]
    np.testing.assert_allclose(V1, V2)
