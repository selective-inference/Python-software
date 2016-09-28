import numpy as np
from initial_soln import selection
from selection.algorithms.lasso import instance
from scipy.stats import norm as ndist
from sel_probability import selection_probability

n=100
p=10
s=3
snr=5

X_1, y, true_beta, nonzero, sigma = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
random_Z = np.random.standard_normal(p)

print true_beta

sel = selection(X_1,y, random_Z)
if sel is not None:
    lam, epsilon, active, betaE, cube, initial_soln = sel
    nactive=betaE.shape[0]
    print nactive, lam
    tau=1
    X_perm=np.zeros((n,p))
    X_perm[:,:nactive]=X_1[:,active]
    X_perm[:,nactive:]=X_1[:,~active]
    X=X_perm
    V=-X
    X_E=X[:,:nactive]
    X_E_comp=X[:,nactive:]
    B_E=np.zeros((p,p))
    B_E[:,:nactive]=np.dot(X.T,X[:,:nactive])
    B_E[:nactive, :nactive]+= epsilon*np.identity(nactive)
    B_E[nactive:, nactive:]=lam*np.identity((p-nactive))
    gamma_E=np.zeros(p)
    gamma_E[:nactive]=lam*np.sign(betaE)

##############langevin random walk to sample from posterior
class langevin_rw(object):
    def __init__(self,initial_state,gradient_map,stepsize):
        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_state),gradient_map,stepsize)
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0, scale=1)

    def __iter__(self):
        return self

    def next(self):
        while True:
            candidate = (self.state
                        + 0.5 * self.stepsize * self.gradient_map(self.state)
                        + self._noise.rvs(self._shape) * self._sqrt_step)
            if not np.all(np.isfinite(self.gradient_map(candidate))):
                print candidate, self._sqrt_step
                self._sqrt_step *= 0.8
            else:
                self.state[:] = candidate
                break

ini = betaE
stepsize = np.true_divide(1,np.sqrt(p))
sel = selection_probability(V, B_E, gamma_E, sigma, tau, lam, y, betaE, cube)

def gradient_map(param):
    return sel.gradient(param, y, 1)

target=langevin_rw(ini,gradient_map,stepsize)
langevin_steps = 10000
burnin = 1000
samples = []
for i in range(langevin_steps):
    target.next()
    if (i>=burnin):
        samples.append(target.state.copy())

#print samples[1:10]
