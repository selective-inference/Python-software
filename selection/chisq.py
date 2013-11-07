import numpy as np
import selection.constraints as C
from scipy.stats import chi

# we use R's chisq

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri
ro.numpy2ri.activate()

def tangent_space(operator, y):
    """
    Return the unit vector $\eta(y)=Ay/\|Ay\|_2$ 
    where $A$ is `operator`. It also forms a basis for the 
    tangent space of the unit sphere at $\eta(y)$.

    Parameters
    ----------

    operator : `np.float((p,q))`
    y : `np.float((q,)`

    Returns
    -------

    eta : `np.float(p)`
        Unit vector that achieves the norm of $Ay$ with $A$ as `operator`. 
    tangent_space : `np.float((p-1,p))`
        An array whose rows form a basis for the tangent space
        of the sphere at `eta`.

    """
    A = operator # shorthand
    soln = np.dot(A, y)
    norm_soln = np.linalg.norm(soln)
    eta = np.dot(A.T, soln) / norm_soln

    p, q = A.shape
    if p > 1:
        tangent_vectors = np.identity(p)[:-1]
        for tv in tangent_vectors:
            tv[:] = tv - np.dot(tv, soln) * soln / norm_soln**2
        return eta, np.dot(tangent_vectors, A)
    else:
        return eta, None
    
def quadratic_test(y, operator, con):
    """
    Perform a quadratic test based on some constraints. 
    """
    A = operator # shortand
    p, q = A.shape

    eta, TA = tangent_space(A, y)
    if TA is not None:
        newcon = C.constraints((con.inequality, 
                                con.inequality_offset),
                               (TA, np.zeros(TA.shape[0])),
                               covariance=con.covariance)
        newcon = newcon.impose_equality()
        P = np.identity(q) - np.dot(np.linalg.pinv(TA), TA)
        eta = np.dot(P, eta)
    else:
        newcon = con.impose_equality()

    Vp, V, Vm, sigma = newcon.pivots(eta, y)[:4]
    Vp = max(0, Vp)
    
    sf = chi.sf

    try:
        pval = chi_pvalue(V, Vp, Vm, sigma, p, method='MC', nsim=10000)
    except:
        pval = ((sf(Vm/sigma, p) - sf(V/sigma,p)) / 
                (sf(Vm/sigma, p) - sf(Vp/sigma,p)))
    return np.clip(pval, 0, 1)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from warnings import warn
    try:
        import statsmodels.api as sm
    except ImportError:
        warn('unable to plot ECDF as statsmodels has not imported')

    def full_sim(L, b, p):
        k, q = L.shape
        A1 = np.random.standard_normal((p,q))
        A2 = L[:p]
        A3 = np.array([np.arange(q)**(i/2.) for i in range(1,4)])

        con = C.constraints((L, b), None)
        
        def sim(A):

            y = C.simulate_from_constraints(con) 
            p = A.shape[0]
            eta, B = tangent_space(A, y)

            newcon = C.constraints((L, b), (B, np.zeros(B.shape[0])))
            newcon = newcon.impose_equality()
            P = np.identity(q) - np.dot(np.linalg.pinv(B), B)
            eta = np.dot(P, eta)
            Vp, V, Vm, sigma = newcon.pivots(eta, y)[:4]

            Vp = max(0, Vp)
            pval = chi_pvalue(V, Vp, Vm, sigma, p, method='MC', nsim=10000)
            return np.clip(pval,0,1)

        return sim(A1), sim(A2), sim(A3)

    nsim = 10000
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


def chi_pvalue(L, Mplus, Mminus, sd, k, method='MC', nsim=1000):
    if k == 1:
        H = []
    else:
        H = [0]*(k-1)
    if method == 'cdf':
        pval = (chi.cdf(Mminus / sd, k) - chi.cdf(L / sd, k)) / (chi.cdf(Mminus / sd, k) - chi.cdf(Mplus / sd, k))
    elif method == 'sf':
        pval = (chi.sf(Mminus / sd, k) - chi.sf(L / sd, k)) / (chi.sf(Mminus / sd, k) - chi.sf(Mplus / sd, k))
    elif method == 'MC':
        pval = Q_0(L / sd, Mplus / sd, Mminus / sd, H, nsim=nsim)
    elif method == 'approx':
        if Mminus < np.inf:
            num = np.log((Mminus / sd)**(k-2) * np.exp(-((Mminus/sd)**2-(L/sd)**2)/2.) - 
                         (L/sd)**(k-2))
            den = np.log((Mminus / sd)**(k-2) * np.exp(-((Mminus/sd)**2-(L/sd)**2)/2.) - 
                         (Mplus/sd)**(k-2) * np.exp(-((Mplus/sd)**2-(L/sd)**2)/2.))
            pval = np.exp(num-den)
        else:
            pval = (L/Mplus)**(k-2) * np.exp(-((L/sd)**2-(Mplus/sd)**2)/2)
    else:
        raise ValueError('method should be one of ["cdf", "sf", "MC"]')
    if pval == 1:
        pval = Q_0(L / sd, Mplus / sd, Mminus / sd, H, nsim=50000)
    if pval > 1:
        pval = 1
    return pval


def q_0(M, Mminus, H, nsim=100):
    Z = np.fabs(np.random.standard_normal(nsim))
    keep = Z < Mminus - M
    proportion = keep.sum() * 1. / nsim
    Z = Z[keep]
    if H != []:
        HM = np.clip(H + M, 0, np.inf)
        exponent = np.log(np.add.outer(Z, HM)).sum(1) - M*Z - M**2/2.
    else:
        exponent = - M*Z - M**2/2.
    C = exponent.max()

    return np.exp(exponent - C).mean() * proportion, C

def Q_0(L, Mplus, Mminus, H, nsim=100):

    exponent_1, C1 = q_0(L, Mminus, H, nsim=nsim)
    exponent_2, C2 = q_0(Mplus, Mminus, H, nsim=nsim)

    return np.exp(C1-C2) * exponent_1 / exponent_2
