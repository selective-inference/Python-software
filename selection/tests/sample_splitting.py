import os
import numpy as np
from scipy.stats import norm as ndist
from selection.covtest import reduced_covtest, covtest
from selection.affine import constraints, gibbs_test
from selection.forward_step import forward_stepwise
n, p, sigma = 50, 80, 1.5

from multi_forward_step import forward_step

def sample_split(X, Y, sigma=None,
                 nstep=10,
                 burnin=1000,
                 ndraw=5000,
                 reduced=True):

    n, p = X.shape
    half_n = int(n/2)
    X1, Y1 = X[:half_n,:]*1., Y[:half_n]*1.
    X2, Y2 = X[half_n:], Y[half_n:]

    FS_half = forward_stepwise(X1, Y1) # sample splitting model
    FS_full = forward_stepwise(X.copy(), Y.copy()) # full data model
    
    spacings_P = []
    split_P = []
    reduced_Pknown = []
    reduced_Punknown = []
    covtest_P = []

    for i in range(nstep):

        FS_half.next()

        if FS_half.P[i] is not None:
            RX = FS_half.X - FS_half.P[i](FS_half.X)
            RY = FS_half.Y - FS_half.P[i](FS_half.Y)
            covariance = np.identity(FS_half.Y.shape[0]) - np.dot(FS_half.P[i].U, FS_half.P[i].U.T)
        else:
            RX = FS_half.X
            RY = FS_half.Y
            covariance = None
        RX -= RX.mean(0)[None,:]
        RX /= (RX.std(0)[None,:] * np.sqrt(RX.shape[0]))

        con, pval, idx, sign = covtest(RX, RY, sigma=sigma,
                                       covariance=covariance,
                                       exact=True)
        covtest_P.append(pval)

        # spacings on half -- not saved

        eta1 = RX[:,idx] * sign
        Acon = constraints(FS_half.A, np.zeros(FS_half.A.shape[0]))
        Acon.covariance *= sigma**2
        Acon.pivot(eta1, FS_half.Y)

        # sample split

        eta2 = np.linalg.pinv(X2[:,FS_half.variables])[-1]
        eta_sigma = np.linalg.norm(eta2) * sigma
        split_P.append(2*ndist.sf(np.fabs((eta2*Y2).sum() / eta_sigma)))

        # inference on full mu using split model, this \beta^+_s.

        zero_block = np.zeros((Acon.linear_part.shape[0], (n-half_n)))
        linear_part = np.hstack([Acon.linear_part, zero_block])
        Fcon = constraints(linear_part, Acon.offset)
        Fcon.covariance *= sigma**2

        if i > 0:
            U = FS_half.P[-2].U.T
            U = np.hstack([U, np.zeros((U.shape[0], (n-half_n)))])
            Uy = np.dot(U, Y)
            Fcon = Fcon.conditional(U, Uy)
        else:
            Fcon = Fcon

        eta_full = np.linalg.pinv(X[:,FS_half.variables])[-1]

        if reduced:
            reduced_pval = gibbs_test(Fcon, Y, eta_full,
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      sigma_known=sigma is not None,
                                      alternative='twosided')[0]
            reduced_Pknown.append(reduced_pval)

            reduced_pval = gibbs_test(Fcon, Y, eta_full,
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      sigma_known=False,
                                      alternative='twosided')[0]
            reduced_Punknown.append(reduced_pval)


        # now use all the data

        FS_full.next()
        if FS_full.P[i] is not None:
            RX = X - FS_full.P[i](X)
            RY = Y - FS_full.P[i](Y)
            covariance = np.identity(RY.shape[0]) - np.dot(FS_full.P[i].U, FS_full.P[i].U.T)
        else:
            RX = X
            RY = Y.copy()
            covariance = None
        RX -= RX.mean(0)[None,:]
        RX /= RX.std(0)[None,:]

        con, pval, idx, sign = covtest(RX, RY, sigma=sigma,
                                       covariance=covariance,
                                       exact=False)
        covtest_P.append(pval)

        # spacings on full data

        eta1 = RX[:,idx] * sign
        Acon = constraints(FS_full.A, np.zeros(FS_full.A.shape[0]))
        Acon.covariance *= sigma**2
        spacings_P.append(Acon.pivot(eta1, Y))

    return split_P, reduced_Pknown, reduced_Punknown, spacings_P, covtest_P, FS_half.variables

def simulation(n, p, sigma, nnz=0, nsim=1000,
               reduced=True,
               reduced_full=True): # nnz = number nonzero

    beta = np.zeros(p)
    if nnz > 0:
        beta[:nnz] = np.linspace(4,4.5,nnz)
    beta = beta * sigma

    splitP = []
    covtestP = []
    spacings = []
    reduced_known = []
    reduced_unknown = []
    reduced_known_full = []
    reduced_unknown_full = []
    hypotheses = []
    hypotheses_full = []

    for i in range(nsim):
        X = (np.random.standard_normal((n,p)) + 
             0.7 * np.random.standard_normal(n)[:,None])
        X -= X.mean(0)[None,:]
        X /= (X.std(0)[None,:] * np.sqrt(n))
        Y = np.random.standard_normal(n) * sigma + np.dot(X, beta)

        split = sample_split(X.copy(),
                             Y.copy(),
                             sigma=sigma,
                             burnin=2000,
                             ndraw=5000,
                             nstep=10,
                             reduced=reduced)
        splitP.append(split[0])
        reduced_known.append(split[1])
        reduced_unknown.append(split[2])
        spacings.append(split[3])
        covtestP.append(split[4])
        hypotheses.append([var in range(nnz) for var in split[5]])

        if reduced_full:
            fs = forward_step(X, Y,
                              sigma=sigma,
                              burnin=2000,
                              ndraw=5000,
                              nstep=10)
            reduced_known_full.append(fs[1])
            reduced_unknown_full.append(fs[2])
            hypotheses_full.append([var in range(nnz) for var in fs[4]])


        for D, name in zip([splitP, spacings, covtestP], ['split', 'spacings', 'covtest']):
            means = map(lambda x: x[~np.isnan(x)].mean(), np.array(D).T)[:(nnz+3)]
            SDs = map(lambda x: x[~np.isnan(x)].std(), np.array(D).T)[:(nnz+3)]
            print means, SDs, name

        if reduced:
            print (np.mean(np.array(reduced_known)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_known)[:,:(nnz+3)],0), 'reduced known split')
            print (np.mean(np.array(reduced_unknown)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_unknown)[:,:(nnz+3)],0), 'reduced unknown split'), i

        if reduced_full:
            print (np.mean(np.array(reduced_unknown_full)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_unknown_full)[:,:(nnz+3)],0), 'reduced unknown full'), i
            print (np.mean(np.array(reduced_known_full)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_known_full)[:,:(nnz+3)],0), 'reduced known full'), i

        value = np.mean(beta)
        if reduced:
            np.save('reduced_split_known%d.npy' % (nnz,), np.array(reduced_known))
            np.save('reduced_split_unknown%d.npy' % (nnz,), np.array(reduced_unknown))

        np.save('split%d.npy' % (nnz,), np.array(splitP))
        np.save('spacings_split%d.npy' % (nnz,), np.array(spacings))
        np.save('covtest_split%d.npy' % (nnz,), np.array(covtestP))
        np.save('hypotheses_split_%d.npy' % (nnz,), np.array(hypotheses))

        if reduced_full:
            np.save('hypotheses_splitfull_%d.npy' % (nnz,), np.array(hypotheses_full))
            np.save('reduced_splitfull_known%d.npy' % (nnz,), np.array(reduced_known_full))
            np.save('reduced_splitfull_unknown%d.npy' % (nnz,), np.array(reduced_unknown_full))
        os.system('cp *split*npy ~/Dropbox/sample_split')

if __name__ == "__main__":

    import sys
    if len(sys.argv) > 1:
        simulation(n, p, sigma, nnz=int(sys.argv[1]), reduced_full=True, reduced=True)
