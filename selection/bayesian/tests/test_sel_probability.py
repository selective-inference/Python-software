import numpy as np
from scipy.optimize import minimize
from tests.instance import gaussian_instance
from selection.tests.instance import gaussian_instance
from bayesian.initial_soln import selection
#from matplotlib import pyplot as plt
#####for debugging currently; need to change this part

n=100
p=10
s=3
snr=5

X_1, y, true_beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)

random_Z = np.random.standard_normal(p)
sel = selection(X_1,y, random_Z)
if sel is not None:
    lam, epsilon, active, betaE, cube, initial_soln = sel
    nactive=betaE.shape[0]
    #print nactive, lam
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

class selection_probability(object):

    #defining class variables
    def __init__(self,V,B_E,gamma_E,sigma,tau,lam,y,betaE,cube):

        (self.V, self.B_E,self.gamma_E,self.sigma,self.tau,self.lam,self.y,self.betaE,self.cube) = (V,B_E,gamma_E,
                                                                                          sigma,tau,lam,y,betaE,cube)
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
        self.Sigma_inter=np.true_divide(np.identity(self.p),self.tau_sq)-np.true_divide(np.dot(np.dot(
            self.V.T,self.Sigma_inv),self.V),self.tau_sq ** 2)
        self.constant = np.true_divide(np.dot(np.dot(self.V_E.T, self.Sigma_inv), self.V_E), self.sigma_sq ** 2)
        self.mat_inter = -np.dot(np.true_divide(np.dot(self.B_E.T, self.V.T), self.tau_sq), self.Sigma_inv)
        self.Sigma_noise = np.dot(np.dot(self.B_E.T, self.Sigma_inter), self.B_E)
        self.vec_inter = np.true_divide(np.dot(self.B_E.T, self.gamma_E), self.tau_sq)
        self.mu_noise = np.dot(self.mat_inter,- np.true_divide(np.dot(self.V, self.gamma_E),
                                                               self.tau_sq)) - self.vec_inter
        self.mu_coef = np.true_divide(-self.lam * np.dot(self.C_E, self.signs), self.tau_sq)
        self.Sigma_coef = np.true_divide(np.dot(self.C_E, self.C_E) + np.dot(self.D_E, self.D_E.T), self.tau_sq)
        self.mu_data = - np.true_divide(np.dot(self.V, self.gamma_E), self.tau_sq)

    # in case of Lasso, the below should return the mean of generative selected model
    #def mean_generative(self,param):
    #    return -np.dot(self.V_E, param)

    # defining log prior to be the Gaussian prior
    def log_prior(self, param, gamma):
        return -np.true_divide(np.linalg.norm(param) ** 2, 2*(gamma ** 2))

    def optimization(self, param):

        # defining barrier function on betaE
        def barrier_sel(z_2):
            # A_betaE beta_E\leq 0
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

        def barrier_subgrad_coord(z):
            if -1+np.power(10, -9)<z<1-np.power(10, -9):
                return np.log(1+np.true_divide(1,1-z))+np.log(1+np.true_divide(1,1+z))
            return 2 * np.log(1 + 10 ** 9)

        def objective_noise(z):

            z_2 = z[:self.nactive]
            z_3 = z[self.nactive:]
            mu_noise_mod = self.mu_noise.copy()
            mu_noise_mod+=np.dot(self.mat_inter,np.true_divide(-np.dot(self.V_E, param), self.sigma_sq))
            return np.true_divide(np.dot(np.dot(z.T, self.Sigma_noise), z), 2)+barrier_sel(
                z_2)+barrier_subgrad(z_3)-np.dot(z.T, mu_noise_mod)

        # defining the objective for subgradient coordinate wise
        def obj_subgrad(z,mu_coord):
            return -(z * mu_coord) + np.true_divide(z ** 2, 2 * self.tau_sq) + barrier_subgrad_coord(z)

        def value_subgrad_coordinate(z_1, z_2):
            mu_subgrad = np.true_divide(-np.dot(self.V_E_comp.T, z_1) - np.dot(self.D_E.T, z_2), self.tau_sq)
            res_seq=[]
            for i in range(self.ninactive):
                mu_coord=mu_subgrad[i]
                res=minimize(obj_subgrad, x0=self.cube[i], args=mu_coord)
                res_seq.append(-res.fun)
            return(np.sum(res_seq))

        def objective_coef(z_2,z_1):
            mu_coef_mod=self.mu_coef.copy()- np.true_divide(np.dot(np.dot(
                self.C_E, self.V_E.T) + np.dot(self.D_E, self.V_E_comp.T), z_1),self.tau_sq)
            return - np.dot(z_2.T,mu_coef_mod) + np.true_divide(np.dot(np.dot(
                z_2.T,self.Sigma_coef),z_2),2)+barrier_sel(z_2)-value_subgrad_coordinate(z_1, z_2)

        # defining objectiv over z_1
        def objective_data(z_1):
            mu_data_mod = self.mu_data.copy()+ np.true_divide(-np.dot(self.V_E, param), self.sigma_sq)
            value_coef = minimize(objective_coef, x0=self.betaE, args=z_1)
            return -np.dot(z_1.T, mu_data_mod) + np.true_divide(np.dot(np.dot(z_1.T, self.Sigma), z_1),
                                                                2) + value_coef.fun

        if self.p< self.n+self.nactive:
            initial_noise = np.zeros(self.p)
            initial_noise[:self.nactive] = self.betaE
            initial_noise[self.nactive:] = self.cube
            res=minimize(objective_noise,x0=initial_noise)
            const_param=np.dot(np.dot(param.T,self.constant),param)
            return -res.fun+const_param, res.x
        else:
            initial_data = self.y
            res = minimize(objective_data, x0=initial_data)
            return -res.fun, res.x

    def selective_map(self,y,prior_sd):
        def objective(param,y,prior_sd):
            return -np.true_divide(np.dot(y.T,-np.dot(self.V_E, param)),
                              self.sigma_sq)-self.log_prior(param,prior_sd)+self.optimization(param)[0]
        map_prob=minimize(objective,x0=self.betaE,args=(y,prior_sd))
        return map_prob.x


sel = selection_probability(V, B_E, gamma_E, sigma, tau, lam, y, betaE, cube)
print sel.selective_map(y,1),np.dot(np.dot(np.dot(X_E.T,X_E),X_E.T),y), true_beta, active
#def objective_map(param):
#    return -np.true_divide(np.dot(y.T, np.dot(X_E, param)),sigma**2)+sel.optimization(param)[0]
#map_prob = minimize(objective_map, x0=betaE)
#print map_prob.x, np.dot(np.dot(np.dot(X_E.T,X_E),X_E.T),y), true_beta, active






#param=5*np.ones(nactive)

#check function mean_generative
#print(np.dot(X[:,:nactive],param)-sel.mean_generative(param))
#print(sel.mean_generative(param))

#check barrier functions
#print sel.optimization(param)(betaE)
#print nactive * np.log(1 + 10 ** 9)
#print sel.optimization(param)(np.append(2,np.random.uniform(-1, 1, (p-nactive-1))))
#print  2*(p-nactive)*np.log(1 + 10 ** 9)

#check optimization when n+nactive> p
#initial_noise = np.sign(betaE)
#initial_noise[:nactive] = betaE
#initial_noise[nactive:] = np.random.uniform(-1, 1,(p-nactive))
#print sel.optimization(param)(initial_noise)

#print sel.optimization(param)(4*np.zeros(n))
#checking shape of selection probability
#print sel.optimization(param)
#if nactive==1:
#    snr_seq = np.linspace(-10, 10, num=20)
#    sel_seq = []
#    for i in range(snr_seq.shape[0]):
     #   print "parameter value", snr_seq[i]
     #   sel = selection_probability(V, B_E, gamma_E, sigma, tau, lam, y, betaE)
     #   sel_int = sel.optimization(snr_seq[i]*np.ones(nactive))
     #   print "log selection probability", sel_int
     #   sel_seq.append(sel_int[0])

    #plt.clf()
    #plt.title("sel_prob")
    #plt.plot(snr_seq, sel_seq)
    #plt.show()

#checking output of value_subgrad
#print sel.optimization(param)(5*np.ones(n),betaE)
#print sel.optimization(param)(5*np.ones(n),betaE)
#checking output of objective_data_coef
#print sel.optimization(param)
#check output of optimization of objective_data_coef


#print sel.selective_map()
#sel = selection_probability(V,B_E,gamma_E,sigma,tau,lam,y,betaE)
#print sel.selective_map()
#print true_beta
#print active
#print np.dot(np.dot(np.dot(X_E.T,X_E),X_E.T),y)

