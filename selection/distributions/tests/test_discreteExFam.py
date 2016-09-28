# Testing
from __future__ import print_function
import numpy as np
import nose.tools as nt
from scipy.stats import poisson
from selection.distributions.discrete_family import discrete_family

def test_discreteExFam():

    X = np.arange(100)
    pois = discrete_family(X, poisson.pmf(X, 1))
    tol = 1e-5

    print(pois._leftCutFromRight(theta=0.4618311,rightCut=(5,.5)), pois._test2RejectsLeft(theta=2.39,observed=5,auxVar=.5))
    print (pois.interval(observed=5,alpha=.05,randomize=True,auxVar=.5))

    print (abs(1-sum(pois.pdf(0))))
    pois.ccdf(0, 3, .4)

    print (pois.Var(np.log(2), lambda x: x))
    print (pois.Cov(np.log(2), lambda x: x, lambda x: x))

    lc = pois._rightCutFromLeft(0, (0,.01))
    print ((0,0.01), pois._leftCutFromRight(0, lc))

    pois._rightCutFromLeft(-10, (0,.01))
    #[pois.test2Cutoffs(t)[1] for t in range(-10,3)]
    pois._critCovFromLeft(-10, (0,.01))

    pois._critCovFromLeft(0, (0,.01))
    pois._critCovFromRight(0, lc)

    pois._critCovFromLeft(5, (5, 1))

    pois._test2RejectsLeft(np.log(5),5)
    pois._test2RejectsRight(np.log(5),5)

    pois._test2RejectsLeft(np.log(20),5)
    pois._test2RejectsRight(np.log(.1),5)

    print (pois._inter2Upper(5,auxVar=.5))
    print (pois.interval(5,auxVar=.5))

