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

#########################################################
#####defining a class for computing selection probability
class selection_probability(object):

    #defining class variables
    def __init__(self, V,B_E,gamma_E,sigma,tau):

        (self.V, self.B_E,self.gamma_E,self.sigma,self.tau) = (V,B_E,gamma_E,sigma,tau)
        self.sigma_sq = self.sigma ** 2
        self.tau_sq = self.tau ** 2
        self.signs = np.sign(self.betaE)
        self.n, self.p = V.shape
        self.nactive = betaE.shape[0]
        self.ninactive=self.p-self.nactive

    def optimization(self,param):

        # defining barrier function on betaE
        def barrier_sel(z):
            # A_1 beta_E\leq 0
            A_1 = np.zeros((self.nactive, self.nactive))
            A_1 = -np.diag(self.signs)
            if all(- np.dot(A_1, z) >= np.power(10, -9)):
                return np.sum(np.log(1 + np.true_divide(1, - np.dot(A_1, z))))
            return self.nactive * np.log(1 + 10 ** 9)

        # defining barrier function on u_{-E}
        def barrier_subgrad(z):

            # A_2 beta_E\leq b
            A_2 = np.zeros(((2 * self.ninactive), (self.ninactive)))
            A_2[:self.ninactive, :] = np.identity(self.ninactive)
            A_2[self.ninactive:, :] = -np.identity(self.ninactive)
            b = np.ones((2 * self.ninactive))
            if all(b - np.dot(A_2, z) >= np.power(10, -9)):
                return np.sum(np.log(1 + np.true_divide(1, b - np.dot(A_2, z))))
            return b.shape[0] * np.log(1 + 10 ** 9)

        Sigma=np.true_divide(np.identity((self.n,self.n)),self.sigma_sq)+np.true_divide(np.dot(self.V,self.V.T),
                                                                                        self.tau_sq)
        Sigma_inv=np.linalg.inv(Sigma)
        Sigma_inter=np.identity(self.p)-np.true_divide(np.dot(np.dot(self.V.T,Sigma_inv),self.V),self.tau_sq ** 2)
        def mean(param):
            return np.dot(-self.V[:,:self.nactive],param)

        mat_inter=np.dot(np.true_divide(np.dot(self.B_E.T, self.V.T), self.tau_sq), Sigma_inv)
        vec_inter=np.true_divide(np.dot(self.B_E.T,self.gamma_E),self.tau_sq)
        mu_tilde=-np.dot(mat_inter,np.true_divide(mean(param),self.sigma_sq)-np.true_divide(np.dot(self.V,self.gamma_E),
                                                                                    self.tau_sq))-vec_inter
        Sigma_tilde=np.dot(np.dot(self.B_E.T,Sigma_inter),self.B_E)

        def objective_1(z):

            z_1=z[1:self.nactive]
            z_2=z[self.nactive:]
            return np.true_divide(np.dot(np.dot(z.T,Sigma_tilde),z),2)+ barrier_sel(z_1)\
                   +barrier_subgrad(z_2)-np.dot(z.T,mu_tilde)

        def




























