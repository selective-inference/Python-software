import numpy as np
from initial_soln import instance, selection
from scipy.optimize import minimize

#####for debugging currently; need to change this part
n=100
p=20
s=5
snr=5
data_instance = instance(n, p, s, snr)
X_1, y, true_beta, nonzero, sigma = data_instance.generate_response()

random_Z = np.random.standard_normal(p)
lam, epsilon, active, betaE, cube, initial_soln = selection(X_1,y, random_Z)

nactive=betaE.shape[0]
tau=1
X_perm=np.zeros((n,p))
X_perm[:,:nactive]=X_1[:,active]
X_perm[:,nactive:]=X_1[:,-active]
X=X_perm
V=-X
B_E=np.zeros((p,p))
B_E[:,:nactive]=np.dot(X.T,X[:,:nactive])
B_E[:nactive, :nactive]+= epsilon*np.identity(nactive)
B_E[nactive:, nactive:]=np.identity((p-nactive))
gamma_E=np.zeros(p)
gamma_E[:nactive]=lam* np.sign(betaE)

class selection_probability(object):

    #defining class variables
    def __init__(self, V,B_E,gamma_E,sigma,tau,lam,y,betaE):

        (self.V, self.B_E,self.gamma_E,self.sigma,self.tau,self.lam,self.y,self.betaE) = (V,B_E,gamma_E,
                                                                                          sigma,tau,lam,y,betaE)
        self.sigma_sq = self.sigma ** 2
        self.tau_sq = self.tau ** 2
        self.signs = np.sign(self.betaE)
        self.n= self.y.shape[0]
        self.p=self.B_E.shape[0]
        self.nactive = self.betaE.shape[0]
        self.ninactive=self.p-self.nactive
        #for lasso, V=-X, B_E=\begin{pmatrix} X_E^T X_E+\epsilon I & 0 \\ X_{-E}^T X_E & I \end{pmatrix}, gamma_E=
        #\begin{pmatrix} \lambda* s_E \\ 0\end{pamtrix}

        # be careful here to permute the active columns beforehand as code
        # assumes the active columns in the first |E| positions
        self.V_E = self.V[:,:self.nactive]
        self.V_E_comp=self.V[:,self.nactive:]
        self.C_E=self.B_E[:self.nactive,:self.nactive]
        self.D_E=self.B_E.T[:self.nactive,self.nactive:]
        self.Sigma = np.true_divide(np.identity(self.n), self.sigma_sq) + np.true_divide(
            np.dot(self.V, self.V.T),self.tau_sq)
        self.Sigma_inv=np.linalg.inv(self.Sigma)
        self.Sigma_inter=np.identity(self.p)-np.true_divide(np.dot(np.dot(self.V.T,self.Sigma_inv),self.V),
                                                            self.tau_sq ** 2)

    # in case of Lasso, the below should return the mean of generative selected model
    def mean_generative(self,param):
        return -np.dot(self.V_E, param)

    def optimization(self, param):
        # defining barrier function on betaE
        def barrier_sel(z_2):
            # A_betaE beta_E\leq 0
            #           # A_betaE = np.zeros((self.nactive, self.nactive))
            A_betaE = -np.diag(self.signs)
            if all(- np.dot(A_betaE, z_2) >= np.power(10, -9)):
                return np.sum(np.log(1 + np.true_divide(1, - np.dot(A_betaE, z_2))))
            return self.nactive * np.log(1 + 10 ** 9)

        def barrier_subgrad(z_3):
            # A_2 beta_E\leq b
            A_subgrad = np.zeros(((2 * self.ninactive), (self.ninactive)))
            A_subgrad[:self.ninactive, :] = np.identity(self.ninactive)
            A_subgrad[self.ninactive:, :] = -np.identity(self.ninactive)
            b = np.ones((2 * self.ninactive))
            if all(b - np.dot(A_subgrad, z_3) >= np.power(10, -9)):
                return np.sum(np.log(1 + np.true_divide(1, b - np.dot(A_subgrad, z_3))))
            return b.shape[0] * np.log(1 + 10 ** 9)

        def objective_noise(z):
            z_2 = z[:self.nactive]
            z_3 = z[self.nactive:]
            #Sigma_noise = np.dot(np.dot(self.B_E.T, self.Sigma_inter), self.B_E)
           # mat_inter = -np.dot(np.true_divide(np.dot(self.B_E.T, self.V.T), self.tau_sq), self.Sigma_inv)
           # vec_inter = np.true_divide(np.dot(self.B_E.T, self.gamma_E), self.tau_sq)
           # mu_noise = np.dot(mat_inter, np.true_divide(self.mean_generative(param), self.sigma_sq) - np.true_divide
            #(np.dot(self.V, self.gamma_E), self.tau_sq)) - vec_inter
            #return np.true_divide(np.dot(np.dot(z.T, Sigma_noise), z), 2) + barrier_sel(z_2) + barrier_subgrad(
            #    z_3) - np.dot(z.T, mu_noise)
            return barrier_sel(z_2)+barrier_subgrad(z_3)

        return objective_noise

        #initial_noise = np.zeros(self.p)
        #initial_noise[:self.nactive] = self.betaE
        #initial_noise[self.nactive:] = np.random.uniform(-1, 1, self.ninactive)
        #res = minimize(objective_noise, x0=initial_noise)
        #return -res.fun




param=np.zeros(nactive)
sel=selection_probability(V,B_E,gamma_E,sigma,tau,lam,y,betaE)
#print(np.dot(X[:,:nactive],param)-sel.mean_generative(param))
#print(sel.mean_generative(param))
#print sel.optimization(param)(betaE)
#print nactive * np.log(1 + 10 ** 9)
#print sel.optimization(param)(np.random.uniform(-1, 1, (p-nactive)))
#print sel.optimization(param)(np.append(2,np.random.uniform(-1, 1, (p-nactive-1))))
#print  2*(p-nactive)*np.log(1 + 10 ** 9)
initial_noise = np.zeros(p)
initial_noise[:nactive] = betaE
initial_noise[nactive:] = np.random.uniform(-1, 1,(p-nactive))
print sel.optimization(param)(initial_noise)
