# Testing
import numpy as np
import nose.tools as nt
from scipy.stats import poisson
import discreteExFam; reload(discreteExFam)
from discreteExFam import DiscreteExFam


X = np.arange(100)
pois = DiscreteExFam(X, poisson.pmf(X, 1))
tol = 1e-5

print (pois.leftCutFromRight(theta=0.4618311,rightCut=(5,.5)), pois.test2RejectsLeft(theta=2.39,x=5,auxVar=.5))
print pois.interval(x=5,alpha=.05,randomize=True,auxVar=.5)

print abs(1-sum(pois.pdf(0)))
pois.ccdf(0, 3, .4)

print pois.Var(np.log(2))
print pois.Cov(np.log(2), lambda x: x, lambda x: x)

lc = pois.rightCutFromLeft(0, (0,.01))
print (0,0.01), pois.leftCutFromRight(0, lc)

pois.rightCutFromLeft(-10, (0,.01))
#[pois.test2Cutoffs(t)[1] for t in range(-10,3)]
pois.critCovFromLeft(-10, (0,.01))

pois.critCovFromLeft(0, (0,.01))
pois.critCovFromRight(0, lc)

pois.critCovFromLeft(5, (5, 1))

pois.test2RejectsLeft(np.log(5),5)
pois.test2RejectsRight(np.log(5),5)

pois.test2RejectsLeft(np.log(20),5)
pois.test2RejectsRight(np.log(.1),5)

print pois.inter2Upper(5,auxVar=.5)
print pois.interval(5,auxVar=.5)

