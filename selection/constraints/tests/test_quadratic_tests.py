import numpy as np
from selection.distributions import chisq 
from scipy.stats import chi
import nose.tools as nt

from selection.tests.decorators import set_sampling_params_iftrue
import selection.constraints.affine as AC

# make any plots not use display

from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt

# used for ECDF

import statsmodels.api as sm

# we use R's chisq

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri
ro.numpy2ri.activate()

@set_sampling_params_iftrue(True)
def test_chisq_central(nsim=None, burnin=8000, ndraw=2000):

    n, p = 4, 10
    A, b = np.random.standard_normal((n, p)), np.zeros(n)
    con = AC.constraints(A,b)

    while True:
        z = np.random.standard_normal(p)
        if con(z):
            break

    S = np.identity(p)[:3]
    Z = AC.sample_from_constraints(con, z, ndraw=ndraw, burnin=burnin)
    P = []
    for i in range(Z.shape[0]/10):
        P.append(chisq.quadratic_test(Z[10*i], S, con))
    ecdf = sm.distributions.ECDF(P)

    plt.clf()
    x = np.linspace(0,1,101)
    plt.plot(x, ecdf(x), c='red')
    plt.plot([0,1],[0,1], c='blue', linewidth=2)
    nt.assert_true(np.fabs(np.mean(P)-0.5) < 0.03)
    nt.assert_true(np.fabs(np.std(P)-1/np.sqrt(12)) < 0.03)
    

@set_sampling_params_iftrue(True)
def test_chisq_noncentral(nsim=1000, burnin=2000, ndraw=8000):

    mu = np.arange(6)
    ncp = np.linalg.norm(mu[:3])**2

    A, b = np.random.standard_normal((4,6)), np.zeros(4)
    con = AC.constraints(A,b, mean=mu)

    ro.r('fncp=%f' % ncp)
    ro.r('f = function(x) {pchisq(x,3,ncp=fncp)}')
    def F(x):
        if x != np.inf:
            return np.array(ro.r('f(%f)' % x))
        else:
            return np.array([1.])

    # find a feasible point

    while True:
        z = np.random.standard_normal(mu.shape)
        if con(z):
            break

    P = []
    for i in range(nsim):
        Z = AC.sample_from_constraints(con, z, ndraw=ndraw, burnin=burnin)
        print i
        u = 0 * Z
        u[:3] = Z[:3] / np.linalg.norm(Z[:3])
        L, V, U = con.pivot(u, Z)[:3]
        if L > 0:
            Ln = L**2
            Un = U**2
            Vn = V**2
        else:
            Ln = 0
            Un = U**2
            Vn = V**2

        if U < 0:
            stop
        P.append(np.array((F(Un) - F(Vn)) / (F(Un) - F(Ln))))

    P = np.array(P).reshape(-1)
    P = P[P > 0]
    P = P[P < 1]

    ecdf = sm.distributions.ECDF(P)

    plt.clf()
    x = np.linspace(0,1,101)
    plt.plot(x, ecdf(x), c='red')
    plt.plot([0,1],[0,1], c='blue', linewidth=2)

@set_sampling_params_iftrue(True)
def main_test(nsim=1000, burnin=None, ndraw=None):


    def full_sim(L, b, p):
        k, q = L.shape
        A1 = np.random.standard_normal((p,q))
        A2 = L[:p]
        A3 = np.array([np.arange(q)**(i/2.) for i in range(1,4)])

        con = AC.constraints((L, b), None)
        
        def sim(A):

            y = C.simulate_from_constraints(con) 
            return quadratic_test(y, np.identity(con.dim),
                                  con)

        return sim(A1), sim(A2), sim(A3)

    P = []

    p, q, k = 4, 20, 6
    L, b = np.random.standard_normal((k,q)), np.ones(k) * 0.2

    for _ in range(nsim):
        P.append(full_sim(L, b, p))
    P = np.array(P)

    ecdf = sm.distributions.ECDF(P[:,0])
    ecdf2 = sm.distributions.ECDF(P[:,1])
    ecdf3 = sm.distributions.ECDF(P[:,2])

    plt.clf()
    plt.plot(ecdf.x, ecdf.y, linewidth=4, color='black', label='Fixed (but random) $A$')
    plt.plot(ecdf2.x, ecdf2.y, linewidth=4, color='purple', label='Selected $A$')
    plt.plot(ecdf3.x, ecdf3.y, linewidth=4, color='green', label='Deterministic $A$')

    plt.plot([0,1],[0,1], linewidth=3, linestyle='--', color='red')
    plt.legend(loc='lower right')
    plt.savefig('chisq.pdf')

    # deterministic 

    L2, b2 = np.identity(q)[:4], np.zeros(4)
    P2 = []
    for _ in range(nsim):
        P2.append(full_sim(L2, b2, 3))
    P2 = np.array(P2)

    ecdf = sm.distributions.ECDF(P2[:,0])
    ecdf2 = sm.distributions.ECDF(P2[:,1])
    ecdf3 = sm.distributions.ECDF(P2[:,2])

    plt.clf()
    plt.plot(ecdf.x, ecdf.y, linewidth=4, color='black', label='Fixed (but random) $A$')
    plt.plot(ecdf2.x, ecdf2.y, linewidth=4, color='purple', label='Selected $A$')
    plt.plot(ecdf3.x, ecdf3.y, linewidth=4, color='green', label='Deterministic $A$')

    plt.plot([0,1],[0,1], linewidth=3, linestyle='--', color='red')
    plt.legend(loc='lower right')
    plt.savefig('chisq_det.pdf')
