import numpy as np
from scipy.optimize import minimize

class no_scale_selection_probability(object):

    # defining class variables
    def __init__(self, V, B_E, gamma_E, sigma, tau, lam, y, betaE, cube):

        (self.V, self.B_E, self.gamma_E, self.sigma, self.tau, self.lam, self.y, self.betaE, self.cube) = (V, B_E,
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
        # for lasso, V=-X, B_E=\begin{pmatrix} X_E^T X_E+\epsilon I & 0 \\ X_{-E}^T X_E & I \end{pmatrix}, gamma_E=
        # \begin{pmatrix} \lambda* s_E \\ 0\end{pamtrix}

        # be careful here to permute the active columns beforehand as code
        # assumes the active columns in the first |E| positions
        self.V_E = self.V[:, :self.nactive]
        self.V_E_comp = self.V[:, self.nactive:]
        self.C_E = self.B_E[:self.nactive, :self.nactive]
        self.D_E = self.B_E.T[:self.nactive, self.nactive:]
        self.Sigma = np.true_divide(np.identity(self.n), self.sigma_sq) + np.true_divide(
            np.dot(self.V, self.V.T), self.tau_sq)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.Sigma_inter = np.true_divide(np.identity(self.p), self.tau_sq) - np.true_divide(np.dot(np.dot(
            self.V.T, self.Sigma_inv), self.V), self.tau_sq ** 2)
        self.constant=np.true_divide(np.dot(np.dot(self.V_E.T, self.Sigma_inv), self.V_E), self.sigma_sq**2)
        self.mat_inter = -np.dot(np.true_divide(np.dot(self.B_E.T, self.V.T), self.tau_sq), self.Sigma_inv)
        self.Sigma_noise = np.dot(np.dot(self.B_E.T, self.Sigma_inter), self.B_E)
        self.vec_inter = np.true_divide(np.dot(self.B_E.T, self.gamma_E), self.tau_sq)
        self.mu_noise = np.dot(self.mat_inter, - np.true_divide(np.dot(self.V, self.gamma_E),
                                                                self.tau_sq)) - self.vec_inter
        self.mu_coef = np.true_divide(-self.lam * np.dot(self.C_E, self.signs), self.tau_sq)
        self.Sigma_coef = np.true_divide(np.dot(self.C_E, self.C_E) + np.dot(self.D_E, self.D_E.T), self.tau_sq)
        self.mu_data = - np.true_divide(np.dot(self.V, self.gamma_E),self.tau_sq)

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
            # A_2 beta_E\leq b
            # a = np.array([1,-1])
            # b = np.ones(2)
            if -1 + np.power(10, -9) < z < 1 - np.power(10, -9):
                return np.log(1 + np.true_divide(1, (1 - z))) + np.log(1 + np.true_divide(1,(1 + z)))
            return 2 * np.log(1 + np.true_divide(10 ** 9,self.lam))

        #defining objective function in p dimensions to be optimized when p<n+|E|
        def objective_noise(z):

            z_2 = z[:self.nactive]
            z_3 = z[self.nactive:]
            mu_noise_mod = self.mu_noise.copy()
            mu_noise_mod+=np.dot(self.mat_inter,np.true_divide(-np.dot(self.V_E, param), self.sigma_sq))
            return np.true_divide(np.dot(np.dot(z.T, self.Sigma_noise), z), 2)+barrier_sel(
                z_2)+barrier_subgrad(z_3)-np.dot(z.T, mu_noise_mod)

        #defining objective in 3 steps when p>n+|E|, first optimize over u_{-E}
        # defining the objective for subgradient coordinate wise
        def obj_subgrad(z, mu_coord):
            return -(self.lam*(z * mu_coord)) + ((self.lam**2)*np.true_divide(z ** 2, 2 * self.tau_sq)) \
                   + barrier_subgrad_coord(z)

        def value_subgrad_coordinate(z_1, z_2):
            mu_subgrad = np.true_divide(-np.dot(self.V_E_comp.T, z_1) - np.dot(self.D_E.T, z_2), self.tau_sq)
            res_seq=[]
            for i in range(self.ninactive):
                mu_coord=mu_subgrad[i]
                res=minimize(obj_subgrad, x0=self.cube[i], args=mu_coord)
                res_seq.append(-res.fun)
            return np.sum(res_seq)

        #defining objective over z_2
        def objective_coef(z_2,z_1):
            mu_coef_mod=self.mu_coef.copy()- np.true_divide(np.dot(np.dot(
                self.C_E, self.V_E.T) + np.dot(self.D_E, self.V_E_comp.T), z_1),self.tau_sq)
            return - np.dot(z_2.T,mu_coef_mod) + np.true_divide(np.dot(np.dot(
                z_2.T,self.Sigma_coef),z_2),2)+ barrier_sel(z_2)-value_subgrad_coordinate(z_1, z_2)

        #defining objective over z_1
        def objective_data(z_1):
            mu_data_mod = self.mu_data.copy()+ np.true_divide(-np.dot(self.V_E, param), self.sigma_sq)
            value_coef = minimize(objective_coef, x0=self.betaE, args=z_1)
            return -np.dot(z_1.T, mu_data_mod) + np.true_divide(np.dot(np.dot(z_1.T, self.Sigma), z_1),
                                                                2) + value_coef.fun

        #if self.p < self.n + self.nactive:
        #    initial_noise = np.zeros(self.p)
        #    initial_noise[:self.nactive] = self.betaE
        #    initial_noise[self.nactive:] = self.cube
        #    res = minimize(objective_noise, x0=initial_noise)
        #    const_param = np.dot(np.dot(param.T,self.constant),param)
        #    return -res.fun+const_param, res.x
        #else:
        initial_data = self.y
        res = minimize(objective_data, x0=initial_data)
        return -res.fun, res.x


        #return objective_data(self.y), value_subgrad_coordinate(self.y, self.betaE)

    def selective_map(self,y,prior_sd):
        def objective(param,y,prior_sd):
            return -np.true_divide(np.dot(y.T,-np.dot(self.V_E, param)),
                              self.sigma_sq)-self.log_prior(param,prior_sd)+self.optimization(param)[0]
        map_prob=minimize(objective,x0=self.betaE,args=(y,prior_sd))
        return map_prob.x

    def gradient(self,param,y,prior_sd):
        if self.p< self.n+self.nactive:
            func_param=np.dot(self.constant,param)
            grad_sel_prob= np.dot(np.dot(self.mat_inter, -np.true_divide(self.V_E, self.sigma_sq)).T,
                                  self.optimization(param)[1])+func_param
        else:
            grad_sel_prob= np.dot(-np.true_divide(self.V_E.T, self.sigma_sq),self.optimization(param)[1])

        return np.true_divide(-np.dot(self.V_E.T,y),self.sigma_sq) -np.true_divide(param,prior_sd**2)-grad_sel_prob