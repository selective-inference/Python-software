# IPython log file

import numpy as np
from selection.covtest import reduced_covtest, covtest
from selection.affine import constraints, gibbs_test
from selection.forward_step import forward_stepwise
n, p, sigma = 50,100,1.5

def forward_step(X, Y, sigma=None,
                 nstep=5,
                 exact=False,
                 burnin=1000,
                 ndraw=5000):
    """
    A simple implementation of forward stepwise
    that uses the `reduced_covtest` iteratively
    after adjusting fully for the selected variable.

    This implementation is not efficient, in
    that it computes more SVDs than it really has to.

    Parameters
    ----------

    X : np.float((n,p))

    Y : np.float(n)

    sigma : float (optional) 
        Noise level (not needed for reduced).

    nstep : int
        How many steps of forward stepwise?

    exact : bool
        Which version of covtest should we use?

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    tests : ['reduced_known', 'covtest', 'reduced_unknown']
        Which test to use? A subset of the above sequence.

    """

    n, p = X.shape
    FS = forward_stepwise(X, Y)

    spacings_P = []
    covtest_P = []
    reduced_Pknown = []
    reduced_Punknown = []

    for i in range(nstep):
        FS.next()

        # covtest
        if FS.P[i] is not None:
            RX = X - FS.P[i](X)
            RY = Y - FS.P[i](Y)
            covariance = np.identity(n) - np.dot(FS.P[i].U, FS.P[i].U.T)
        else:
            RX = X
            RY = Y
            covariance = None
        RX -= RX.mean(0)[None,:]
        RX /= RX.std(0)[None,:]

        con, pval, idx, sign = covtest(RX, RY, sigma=sigma,
                                       covariance=covariance,
                                       exact=exact)
        covtest_P.append(pval)

        # reduced

        eta = RX[:,idx] * sign
        Acon = constraints(FS.A, np.zeros(FS.A.shape[0]))
        Acon.covariance *= sigma**2
        if i > 0:
            U = FS.P[-2].U.T
            Uy = np.dot(U, Y)
            Bcon = Acon.conditional(U, Uy)
        else:
            Bcon = Acon

        spacings_P.append(Acon.pivot(eta, Y))

        reduced_pval, _, _ = gibbs_test(Bcon, Y, eta,
                                        ndraw=ndraw,
                                        burnin=burnin,
                                        sigma_known=sigma is not None,
                                        alternative='greater')
        reduced_Pknown.append(reduced_pval)

        reduced_pval, _, _ = gibbs_test(Bcon, Y, eta,
                                        ndraw=ndraw,
                                        burnin=burnin,
                                        sigma_known=False,
                                        alternative='greater')
        reduced_Punknown.append(reduced_pval)

    return covtest_P, reduced_Pknown, reduced_Punknown, spacings_P, FS.variables

def simulation(n, p, sigma, nnz=0, value=4, nsim=1000): # nnz = number nonzero

    beta = np.zeros(p)
    beta[:nnz] = value * sigma

    spacings = []
    reduced_known = []
    reduced_unknown = []
    covtest = []
    hypotheses = []

    for i in range(nsim):
        X = (np.random.standard_normal((n,p)) + 
             0.7 * np.random.standard_normal(n)[:,None])
        X -= X.mean(0)[None,:]
        X /= X.std(0)[None,:] * np.sqrt(n)
        Y = np.random.standard_normal(n) * sigma + np.dot(X, beta)
        fs = forward_step(X,
                          Y,
                          sigma=sigma,
                          burnin=2000,
                          ndraw=5000,
                          nstep=10)
        covtest.append(fs[0])
        reduced_known.append(fs[1])
        reduced_unknown.append(fs[2])
        spacings.append(fs[3])
        hypotheses.append([var in range(nnz) for var in fs[4]])
#        print (np.mean(np.array(covtest)[:,:(nnz+3)],0), 
#               np.std(np.array(covtest)[:,:(nnz+3)],0), 'cov')
#        print (np.mean(np.array(reduced_known)[:,:(nnz+3)],0), 
#               np.std(np.array(reduced_known)[:,:(nnz+3)],0), 'reduced')
        print (np.mean(np.array(reduced_unknown)[:,:(nnz+3)],0), 
               np.std(np.array(reduced_unknown)[:,:(nnz+3)],0), 'reduced unknown'), i

        np.save('reduced_known%d_%0.1f.npy' % (nnz, value), np.array(reduced_known))
        np.save('reduced_unknown%d_%0.1f.npy' % (nnz, value), np.array(reduced_unknown))
        np.save('covtest%d_%0.1f.npy' % (nnz, value), np.array(covtest))
        np.save('spacings%d_%0.1f.npy' % (nnz, value), np.array(spacings))
        np.save('hypotheses%d_%0.1f.npy' % (nnz, value), np.array(hypotheses))

if __name__ == "__main__":

    import sys
    simulation(n, p, sigma, int(sys.argv[1]))

