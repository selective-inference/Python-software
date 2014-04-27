# IPython log file

import numpy as np
from selection.covtest import forward_step, reduced_covtest
n, p, sigma = 50,100,1.5

def simulation(n, p, sigma, nnz=0, value=4, nsim=1000): # nnz = number nonzero

    beta = np.zeros(p)
    beta[:nnz] = value * sigma
    reduced = []
    covtest = []
    single_step = []
    for i in range(nsim):
        print i
        X = np.random.standard_normal((n,p)) + 0.7 * np.random.standard_normal(n)[:,None]
        X -= X.mean(0)[None,:]
        X /= X.std(0)[None,:] * np.sqrt(n)
        Y = np.random.standard_normal(n) * sigma + np.dot(X, beta)
        fs = forward_step(X,
                          Y,
                          sigma=sigma,
                          burnin=5000,
                          ndraw=20000,
                          nstep=10)
        reduced.append(fs[1])
        covtest.append(fs[0])
        print (np.mean(np.array(covtest)[:,:(nnz+1)],0), 
               np.std(np.array(covtest)[:,:(nnz+1)],0), 'cov')
        print (np.mean(np.array(reduced)[:,:(nnz+1)],0), 
               np.std(np.array(reduced)[:,:(nnz+1)],0), 'reduced')

    np.save('reduced%d_%0.1f.npy' % (nnz, value), np.array(reduced))
    np.save('covtest%d_%0.1f.npy' % (nnz, value), np.array(covtest))

simulation(n,p,sigma,0)
simulation(n,p,sigma,1)
simulation(n,p,sigma,2)
simulation(n,p,sigma,5)
