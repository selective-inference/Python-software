import numpy as np
from initial_soln import selection
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance as instance
#from matplotlib import pyplot as plt
#####for debugging currently; need to change this part

n=100
p=10
s=3
snr=5

X_1, y, true_beta, nonzero, sigma = instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

random_Z = np.random.standard_normal(p)
sel = selection(X_1,y, random_Z)
if sel is not None:
    lam, epsilon, active, betaE, cube, initial_soln = sel
    nactive=betaE.shape[0]
    tau=1
    X_perm=np.zeros((n,p))
    X_perm[:,:nactive]=X_1[:,active]
    X_perm[:,nactive:]=X_1[:,~active]
    X=X_perm
    X_E=X[:,:nactive]
    X_E_comp=X[:,nactive:]
    B_E=np.zeros((p,p))
    B_E[:,:nactive]=np.dot(X.T,X[:,:nactive])
    B_E[:nactive, :nactive]+= epsilon*np.identity(nactive)
    B_E[nactive:, nactive:]=lam*np.identity((p-nactive))
    gamma_E=np.zeros(p)
    gamma_E[:nactive]=lam*np.sign(betaE)

class appoximate_confidence(object):

    # defining class variables
    def __init__(self, X, B_E, gamma_E, sigma, tau, lam, y, betaE, cube):

        (self.X, self.B_E, self.gamma_E, self.sigma, self.tau, self.lam, self.y, self.betaE, self.cube) = (X, B_E,
                                                                                                           gamma_E,
                                                                                                           sigma, tau,
                                                                                                           lam, y,betaE,
                                                                                                           cube)
        self.sigma_sq = self.sigma ** 2
        self.tau_sq = self.tau ** 2
        self.signs = np.sign(self.betaE)
        self.n = self.y.shape[0]
        self.p = self.B_E.shape[0]
        self.nactive = self.betaE.shape[0]
        self.ninactive = self.p - self.nactive
        self.XE_pinv = np.linalg.pinv(self.X[:,:self.active])
        self.vec_inter_1=np.dot(self.B_E.T,self.X.T)
        self.mat_inter_2=np.true_divide(np.dot(self.B_E.T,self.B_E),self.tau_sq)
        self.initial=np.append(self.betaE,self.cube)

        self.eta_norm_sq = np.zeros(self.nactive)
        for j in range(self.nactive):
            eta = self.XE_pinv[j, :]
            self.eta_norm_sq[j] = np.linalg.norm(eta) ** 2

    def optimization(self,j,s):
        eta = self.XE_pinv[j, :]
        c = np.true_divide(eta, self.eta_norm_sq[j])
        fixed_part = np.dot(np.identity(self.n) - np.outer(c, eta), self.y)
        gamma = self.gamma_E.copy()-np.dot(self.X.T, fixed_part)
        mu_noise=np.true_divide((s*np.dot(self.vec_inter,c)-np.dot(self.B_E.T,gamma)),self.tau_sq)

        def barrier(z):
            A = np.zeros((self.p + self.ninactive, self.p))
            A[:self.nactive, :self.nactive] = -np.diag(self.signs)
            A[self.nactive:self.p, self.nactive:] = np.identity(self.ninactive)
            A[self.p:, self.nactive:] = -np.identity(self.ninactive)
            b = np.zeros(self.p + self.ninactive)
            b[self.nactive:] = 1

            if all(b - np.dot(A, z) >= np.power(10, -9)):
                return np.sum(np.log(1 + np.true_divide(1, b - np.dot(A, z))))

            return b.shape[0] * np.log(1 + 10 ** 9)


        def objective_noise(z):
            return -np.dot(z.T,mu_noise)+np.true_divide(np.dot(np.dot(z.T,self.mat_inter),z),2)+barrier(z)

        res=minimize(objective_noise,x0=self.initial)
        return -np.true_divide(np.linalg.norm((s*np.dot(self.X.T,c))-gamma),2*self.tau_sq)-res.fun

    def approx_density(self,s):
        return 





