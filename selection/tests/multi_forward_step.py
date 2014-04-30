# IPython log file

import numpy as np
from selection.covtest import forward_step, reduced_covtest
n, p, sigma = 50,100,1.5

def simulation(n, p, sigma, nnz=0, value=4, nsim=1000): # nnz = number nonzero

    beta = np.zeros(p)
    beta[:nnz] = value * sigma
    reduced_known = []
    reduced_unknown = []
    covtest = []
    single_step = []
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

#        print (np.mean(np.array(covtest)[:,:(nnz+3)],0), 
#               np.std(np.array(covtest)[:,:(nnz+3)],0), 'cov')
#        print (np.mean(np.array(reduced_known)[:,:(nnz+3)],0), 
#               np.std(np.array(reduced_known)[:,:(nnz+3)],0), 'reduced')
        print (np.mean(np.array(reduced_unknown)[:,:(nnz+3)],0), 
               np.std(np.array(reduced_unknown)[:,:(nnz+3)],0), 'reduced unknown')

    np.save('reduced_known%d_%0.1f.npy' % (nnz, value), np.array(reduced_known))
    np.save('reduced_unknown%d_%0.1f.npy' % (nnz, value), np.array(reduced_unknown))
    np.save('covtest%d_%0.1f.npy' % (nnz, value), np.array(covtest))

simulation(n,p,sigma,0)
simulation(n,p,sigma,2)
simulation(n,p,sigma,1)
simulation(n,p,sigma,5)
