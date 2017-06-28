import numpy as np
from selection.distributions import chisq 
from scipy.stats import chi
import nose.tools as nt
import numpy.testing.decorators as dec

from selection.tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
import selection.constraints.affine as AC


# we use R's chisq

try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects.numpy2ri import numpy2ri
    ro.conversion.py2ri = numpy2ri
    ro.numpy2ri.activate()
    R_available = True
except ImportError:
    R_available = False

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=20000)
@set_seed_iftrue(SET_SEED)
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
    for i in range(int(Z.shape[0]/10)):
        P.append(chisq.quadratic_test(Z[10*i], S, con))

    nt.assert_true(np.fabs(np.mean(P)-0.5) < 0.03)
    nt.assert_true(np.fabs(np.std(P)-1/np.sqrt(12)) < 0.03)
    

@dec.skipif(not R_available, "needs rpy2")
@set_sampling_params_iftrue(SMALL_SAMPLES, nsim=10, burnin=10, ndraw=10)
@set_seed_iftrue(SET_SEED)
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
        u = Z[-1]
        u[:3] = u[:3] / np.linalg.norm(u[:3])
        L, V, U = con.bounds(u, Z[-1])[:3]
        if L > 0:
            Ln = L**2
            Un = U**2
            Vn = V**2
        else:
            Ln = 0
            Un = U**2
            Vn = V**2

        P.append(np.array((F(Un) - F(Vn)) / (F(Un) - F(Ln))))

    P = np.array(P).reshape(-1)
    P = P[P > 0]
    P = P[P < 1]


@set_sampling_params_iftrue(SMALL_SAMPLES, nsim=10)
@set_seed_iftrue(SET_SEED)
def main(nsim=1000):


    def full_sim(L, b, p):
        k, q = L.shape
        A1 = np.random.standard_normal((p,q))
        A2 = L[:p]
        A3 = np.array([np.arange(q)**(i/2.) for i in range(1,4)])

        con = AC.constraints(L, b)
        
        def sim(A):

            while True:
                y = np.random.standard_normal(L.shape[1])
                if con(y):
                    break
            return chisq.quadratic_test(y, np.identity(con.dim),
                                        con)

        return sim(A1), sim(A2), sim(A3)

    P = []

    p, q, k = 4, 20, 6
    L, b = np.random.standard_normal((k,q)), np.ones(k) * 0.2

    for _ in range(nsim):
        P.append(full_sim(L, b, p))
    P = np.array(P)

    # make any plots not use display

    from matplotlib import use
    use('Agg')
    import matplotlib.pyplot as plt

    # used for ECDF

    import statsmodels.api as sm

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
