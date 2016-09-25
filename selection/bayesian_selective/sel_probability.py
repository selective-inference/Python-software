import numpy as np
from initial_soln import instance, selection
from scipy.optimize import minimize

#####for debugging currently; need to change this part
n=100
p=20
s=5
snr=5
data_instance = instance(n, p, s, snr)
X, y, true_beta, nonzero, sigma = data_instance.generate_response()

random_Z = np.random.standard_normal(p)
lam, epsilon, active, betaE, cube, initial_soln = selection(X,y, random_Z)

nactive=betaE.shape[0]
X_perm=np.zeros((n,p))
X_perm[:,:nactive]=X[:,active]
X_perm[:,nactive:]=X[:,-active]
X=X_perm
V=-X
B_E=np.zeros((p,p))
B_E[,:nactive]=np.dot(X.T,X[,:nactive])
B_E[:nactive,:nactive]+=+epsilon*np.identity(nactive)
B_E[nactive:, nactive:]=np.identity((p-nactive))
gamma_E=np.zeros(p)
gamma_E[:nactive]=lam* np.signs(beta_E)

#########################################################
#####defining a class for computing selection probability
class selection_probability(object):

    #defining class variables
    def __init__(self, V,B_E,gamma_E,sigma,tau,lam,y,betaE,subgrad):

        (self.V, self.B_E,self.gamma_E,self.sigma,self.tau,self.lam,self.y,self.betaE,self.subgrad) = (V,
                                                                                                       B_E,gamma_E,
                                                                                                       sigma,tau,lam,y,
                                                                                                       betaE,subgrad)
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


    def optimization(self,param):

        # defining barrier function on betaE
        def barrier_sel(z_2):
            # A_betaE beta_E\leq 0
#           # A_betaE = np.zeros((self.nactive, self.nactive))
            A_betaE = -np.diag(self.signs)
            if all(- np.dot(A_betaE, z_2) >= np.power(10, -9)):
                return np.sum(np.log(1 + np.true_divide(1, - np.dot(A_betaE, z_2))))
            return self.nactive * np.log(1 + 10 ** 9)

        # defining barrier function on u_{-E}
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

            z_2=z[:self.nactive]
            z_3=z[self.nactive:]
            Sigma_noise = np.dot(np.dot(self.B_E.T, self.Sigma_inter), self.B_E)
            mat_inter = -np.dot(np.true_divide(np.dot(self.B_E.T, self.V.T), self.tau_sq), self.Sigma_inv)
            vec_inter = np.true_divide(np.dot(self.B_E.T, self.gamma_E), self.tau_sq)
            mu_noise = np.dot(mat_inter, np.true_divide(self.mean_generative(param), self.sigma_sq) - np.true_divide
            (np.dot(self.V, self.gamma_E), self.tau_sq)) - vec_inter
            return np.true_divide(np.dot(np.dot(z.T,Sigma_noise),z),2)+ barrier_sel(z_2)\
                   +barrier_subgrad(z_3)-np.dot(z.T,mu_noise)

        def objective_subgrad(z_3,z_1,z_2):

            mu_subgrad=np.true_divide(-np.dot(self.V_E_comp.T,z_1)-np.dot(self.D_E.T,z_2),self.tau_sq)
            return -np.dot(z_3.T,mu_subgrad)+np.true_divide(np.inner(z_3.T,z_3),2*self.tau_sq)+barrier_subgrad(z_3)

        def value_subgrad(z_1,z_2):
            initial_subgrad=np.random.uniform(-1, 1, self.ninactive)
            res = minimize(objective_subgrad, x0=initial_subgrad,args=(z_1,z_2))
            return -res.fun

        def objective_data_coef(z):
            z_1=z[:self.n]
            z_2=z[self.n:]
            Sigma_coef=np.true_divide(np.dot(self.C_E,self.C_E)+np.dot(self.D_E,self.D_E.T),self.tau_sq)
            mu_coef=np.true_divide(-self.lam*np.dot(self.C_E,self.signs)-np.dot(np.dot(self.C_E,self.V_E.T)+np.dot(
                self.D_E,self.V_E_comp.T),z_1),self.tau_sq)
            mu_data=np.true_divide(self.mean_generative(param),self.sigma_sq) - \
                    np.true_divide(np.dot(self.V, self.gamma_E),self.tau_sq)
            return -np.dot(z_1.T,mu_data)+np.true_divide(np.dot(np.dot(z_1.T,self.Sigma),z_1),2)-value_subgrad(z_1,z_2)-\
                   np.dot(z_2.T,mu_coef)+np.true_divide(np.dot(np.dot(z_2.T,Sigma_coef),z_2),2)+barrier_sel(z_2)

        if self.p< self.n+self.nactive:
            initial_noise = np.zeros(self.p)
            initial_noise[:self.nactive] = self.betaE
            initial_noise[self.nactive:] = np.random.uniform(-1, 1, self.ninactive)
            res=minimize(objective_noise,x0=initial_noise)
            return -res.fun
        else:
            initial_data_coef=np.zeros(self.nactive+self.n)
            initial_data_coef[self.n:]=self.betaE
            res=minimize(objective_data_coef,x0=initial_data_coef)
            return -res.fun















































