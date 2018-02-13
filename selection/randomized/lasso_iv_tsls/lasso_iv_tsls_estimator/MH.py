import regreg.api as rr
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

from selection.randomized.api import randomization as rm
from scipy.stats import laplace


##########################################################################

# sample tsls ESTIMATOR via linear decomposition method, asymptotic method

# keep coordinates in their original places, don't reorganize

##########################################################################

def solve(Z, D, Y, 
          lagrange, randomizer, epsilon, 
          solve_args={'min_its':20, 'tol':1.e-8}):

    n, p = Z.shape
    
    # Set up loss function
    P_Z = Z.dot(np.linalg.pinv(Z))
    #design = np.hstack([P_Z.dot(D).reshape((-1,1)),Z])
    design = np.hstack([Z,P_Z.dot(D).reshape((-1,1))])
    response = P_Z.dot(Y)
    loss = rr.squared_error(design, response)
    
    # Penalty term setup. Feature weights are for the lasso penalization. 
    # The first one is assigned zero because D is not penalized.
    feature_weights = np.ones(1+p) 
    #feature_weights[0] = 0 
    feature_weights[-1] = 0 
    penalty = rr.weighted_l1norm(feature_weights, lagrange=lagrange)
    
    # Randomization and quadratic term, e/2*\|x|_2^2, term setup (x = parameters penalized)
    random_linear_term = randomizer.sample()
    # rr.identity_quadratic essentially amounts to epsilon/2 * \|x - 0\|^2 + <-random_linear_term, x> + 0
    loss.quadratic = rr.identity_quadratic(epsilon, 0, -random_linear_term, 0)

    # Optimization problem   
    problem = rr.simple_problem(loss, penalty)
    problem.solve(**solve_args) 
    soln = problem.coefs
    
    # Extract estimates, active set (i.e. E on alpha only!), and u_{-E} == u_{nE} (on alpha only!)
    #beta = soln[0]
    #E = (soln[1:] != 0)
    #s_E = np.sign(soln[1:])[E]
    #alpha_E = soln[1:][E]
    beta = soln[-1]
    E = (soln[:-1] != 0)
    s_E = np.sign(soln[:-1])[E]
    alpha_E = soln[:-1][E]
    
    # Compute u_{-E} = u_{nE}, 
    # This is from the KKT condition -Z_{-E}^T (Y - Z_{E} alpha_E - D beta) + \lambda u_nE - w_{-E} = 0
    #residual = Y - Z[:,E].dot(alpha_E) - D*beta
    #u_nE = (Z[:,~E].T.dot(residual) + random_linear_term[:-1][~E]) / lagrange
    residual = random_linear_term - (design.T.dot(design)+epsilon*np.identity(p+1)).dot(soln) + design.T.dot(response)
    u_nE = residual[:-1][~E] / lagrange

    return E, s_E, beta, alpha_E, u_nE


def bigaussian_instance(n=1000,p=10,
                        s=3,snr=7.,random_signs=False, #true alpha parameter
                        gsnr = 1., #true gamma parameter
                        beta = 1., #true beta parameter
                        Sigma = np.array([[1., 0.8], [0.8, 1.]]), #noise variance matrix
                        rho=0,scale=False,center=True): #Z matrix structure, note that scale=TRUE will simulate weak IV case!

    # Generate parameters
    # --> Alpha coefficients
    alpha = np.zeros(p) 
    alpha[:s] = snr 
    if random_signs:
        alpha[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True
    # --> gamma coefficient
    gamma = np.repeat([gsnr],p)

    # Generate samples
    # Generate Z matrix 
    Z = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
        np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    if center:
        Z -= Z.mean(0)[None,:]
    if scale:
        Z /= (Z.std(0)[None,:] * np.sqrt(n))
    #    Z /= np.sqrt(n)
    # Generate error term
    mean = [0, 0]
    errorTerm = np.random.multivariate_normal(mean,Sigma,n)
    # Generate D and Y
    D = Z.dot(gamma) + errorTerm[:,1]
    Y = Z.dot(alpha) + D * beta + errorTerm[:,0]
    
    return Z, D, Y, alpha, beta, gamma


class MH_tsls(object):

    def __init__(self, 
                 Z, D, Y, 
                 lagrange, randomizer, epsilon,
                 E, s_E, 
                 beta, alpha_E, u_nE, 
                 b0, Sigmab0,
                 randomization_sampling,
                 data_stepsize=1.,
                 coefs_stepsize=1.5):
        
        n, p = Z.shape
        self.n = n
        self.p = p

        self.lagrange = lagrange
        self.randomizer = randomizer
        self.epsilon = epsilon
        
        self.E = E
        self.s_E = s_E
        self.sizeE = np.sum(E)
    
        self.b0 = b0
        self.Sigmab0 = Sigmab0

        # Store frequent Z projection operations in the sampler
        self.D = D
        self.Y = Y
        self.Z = Z
        ZTZ = Z.T.dot(Z)
        self.ZTZ_Inv = np.linalg.inv(ZTZ)
        P_Z = Z.dot(np.linalg.pinv(Z))
        P_ZE = Z[:,E].dot(np.linalg.pinv(Z[:,E]))
        self.P_diff = P_Z - P_ZE

        # compute observed tsls estimator beta_hat
        self.T_obs = D.dot(self.P_diff).dot(Y) / D.dot(self.P_diff).dot(D)
        self.T_var = self.Sigmab0[0,0] / D.dot(self.P_diff).dot(D)

        # compute K constants
        # K1 = \Sigma_{S,T}
        self.K1 = np.hstack([Z.T.dot(self.P_diff).dot(D).reshape(1,-1), np.atleast_2d(D.dot(self.P_diff).dot(D))])
        self.K1 *= self.Sigmab0[0,0] / (D.dot(self.P_diff).dot(D))
        self.K2 = np.vstack([np.hstack([ZTZ,Z.T.dot(D).reshape(-1,1)]), np.hstack([D.dot(Z), D.dot(P_Z).dot(D)])])+self.epsilon*np.identity(self.p+1)
        self.K3 = np.vstack([np.identity(p),D.dot(Z).dot(self.ZTZ_Inv)]).dot(Z.T.dot(Y)).reshape(1,-1) - self.K1 * self.T_obs

        self.state = np.hstack([self.T_obs,alpha_E,beta,u_nE])

        #print 'initializations: ', self.T_obs, self.K1, self.K2, self.K3
        #print 'initial state:', self.state
        
        self.state_Tindex = slice(0,1,1) #these are numerical indices, different than boolean index E
        self.state_alpha_Eindex = slice(1,1+self.sizeE,1)
        self.state_betaindex = slice(1+self.sizeE,2+self.sizeE,1)
        self.state_u_nEindex = slice(2+self.sizeE,self.p+2,1)   

        self.data_slice = slice(0,1,1)
        self.coefs_slice = slice(1,2+self.sizeE,1)

        # Store the number of total and accepted steps for MH
        self.accept_data = 0
        self.total_data = 0
        self.accept_coefs = 0
        self.total_coefs = 0

        # used in sampling
        self.randomization = randomization_sampling

        # stepsize for proposal_data, proposal_coefs
        self.data_stepsize = data_stepsize / np.sqrt(self.n)
        #self.coefs_stepsize = coefs_stepsize / np.sqrt(self.sizeE + 1)
        self.coefs_stepsize = coefs_stepsize / np.sqrt(self.n)

    def log_pdf_data(self, state):
        # here tsls estimator ~ N(\beta^*, Sigma[0,0]/D^T(P_Z-PZE)D)
        # log(f) part up to some constant
        data = state[self.data_slice]
        return (-(data-self.b0)**2/(2*self.T_var))

    def log_pdf_randomization(self, state):
        return self.randomizer.log_density(self.reconstruction_map(state))

    def log_jacobian(self, state):
        return 0.

    def logpdf(self, state):
        # calculate the entire selective density
        return (self.log_pdf_data(state) + self.log_pdf_randomization(state) + self.log_jacobian(state))

    def step_data(self):
        self.total_data += 1

        data = self.state[self.data_slice].copy()
        data_proposal = data + self.data_stepsize * np.sqrt(self.T_var) * np.random.standard_normal(1)
        state_proposal = self.state.copy()
        state_proposal[self.data_slice] = data_proposal
        log_ratio = self.logpdf(state_proposal) - self.logpdf(self.state)

        if np.log(np.random.uniform()) < log_ratio:
            self.accept_data += 1
            data = data_proposal
        #print 'data', data
        return data

    def step_coefs(self):
        self.total_coefs += 1

        coefs = self.state[self.coefs_slice].copy()
        # should not change the sign of coefs! could probably return this
        coefs_abs = coefs.copy()
        coefs_abs[:-1] = np.abs(coefs_abs[:-1])

        # FUTURE: use the randomization from scipy, may need to fix        
        coefs_proposal = coefs_abs + self.coefs_stepsize * self.randomization.rvs(self.sizeE + 1)
        coefs_proposal[:-1] = np.abs(coefs_proposal[:-1]) * self.s_E
        state_proposal = self.state.copy()
        state_proposal[self.coefs_slice] = coefs_proposal
        log_ratio = self.log_pdf_randomization(state_proposal) - self.log_pdf_randomization(self.state)

        if np.log(np.random.uniform()) < log_ratio:
            self.accept_coefs += 1
            coefs = coefs_proposal
        #print 'coefs', coefs
        return coefs       

    def step_subgrad(self):
        T = self.state[self.state_Tindex]
        alpha_E = self.state[self.state_alpha_Eindex]
        beta = self.state[self.state_betaindex]
        u_nE = self.state[self.state_u_nEindex]

        coefs = np.zeros(1+self.p)
        coefs[:-1][self.E] = alpha_E
        coefs[-1] = beta
        offset = (- self.K1 * T + self.K2.dot(coefs) - self.K3)[0][:-1][~self.E]
        lower = offset - self.lagrange
        upper = offset + self.lagrange
        #print 'lower, upper: ', lower, upper
        percentile = np.random.sample(self.p - self.sizeE) * (self.randomization.cdf(upper) - self.randomization.cdf(lower)) + self.randomization.cdf(lower)
        #print 'percentile', percentile
        #print 'ppf', self.randomization.ppf(percentile)
        subgrad_sample = (self.randomization.ppf(percentile) - offset) / self.lagrange
        return subgrad_sample            

    def reconstruction_map(self, state):
        # Extract elements from current state of sampler
        T = state[self.state_Tindex]
        alpha_E = state[self.state_alpha_Eindex]
        beta = state[self.state_betaindex]
        u_nE = state[self.state_u_nEindex]
        
        # Compute omega. 
        omega = np.zeros(self.p + 1)
        coefs = np.zeros(self.p + 1)
        coefs[:-1][self.E] = alpha_E
        coefs[-1] = beta
        subgrad = np.zeros(self.p + 1)
        subgrad[:-1][self.E] = self.s_E
        subgrad[:-1][~self.E] = u_nE
        omega = - self.K1 * T + self.K2.dot(coefs) + self.lagrange * subgrad - self.K3
        return omega


class MH_sampler(MH_tsls):

    def sample(self, ndraw = 500, burnin = 100, returnAllSamples = False):
        """
        The function samples the distribution of the tsls estimator
        subject to the selection constraints.
        if returnAllSamples, return the entire state samples for diagnostic plots NOT including the burnin
        """
        samplesAll = []
        if returnAllSamples:
            for i in range(ndraw + burnin):
                sample = self.next()
                if i >= burnin:
                    samplesAll.append(sample.copy())
            samplesAll = np.array(samplesAll)
            print "Acceptance Rates for (Data, Coef):", self.accept_data *1./ self.total_data, self.accept_coefs * 1. / self.total_coefs

            if(np.amax(np.absolute(samplesAll[:, self.state_u_nEindex])) > 1) :
                print "--> u_nE max norm constraint not satisfied in sampler :("
            if np.sum(samplesAll[:, self.state_alpha_Eindex].dot(np.diag(self.s_E))<0) > 0:
                print "--> alpha_E does not obey sign constraints in sampler :(" 
            return samplesAll

        else:
            for i in range(ndraw + burnin):
                sample = self.next()
                if i >= burnin:
                    samplesAll.append(sample.copy())
            samplesAll = np.array(samplesAll)
            print "Acceptance Rates for (Data, Coef):", self.accept_data *1./ self.total_data, self.accept_coefs * 1. / self.total_coefs

            if(np.amax(np.absolute(samplesAll[:, self.state_u_nEindex])) > 1) :
                print "--> u_nE max norm constraint not satisfied in sampler :("
            if np.sum(samplesAll[:, self.state_alpha_Eindex].dot(np.diag(self.s_E))<0) > 0:
                print "--> alpha_E does not obey sign constraints in sampler :(" 

            samples = samplesAll[:, self.data_slice]
            return np.array(samples)

    def __iter__(self):
        return self

    def next(self):
        self.state[self.data_slice] = self.step_data()
        self.state[self.coefs_slice] = self.step_coefs()
        self.state[self.state_u_nEindex] = self.step_subgrad()
        return self.state


# null pvalue is when H0: b0 = beta_star the true value
# if not true model, then nuisance estimators use beta_tsls instead of beta_star
def null_pvalue(snr, gsnr, beta_star, Sigma, ndraw, burnin, data_step, coefs_step, true_model=True):

    Z, D, Y, alpha, beta_star, gamma  = bigaussian_instance(snr = snr, gsnr = gsnr, beta = beta_star, Sigma = Sigma)

    n, p = Z.shape
    lagrange = 2.01 * np.sqrt(n * np.log(n))
    #lagrange = 3.
    epsilon = 1. * np.sqrt(n)
    #epsilon = 1.
    #randomizer = rm.isotropic_gaussian((p+1,), scale=0.5*np.sqrt(n))
    randomizer = rm.laplace((p+1,), scale=0.5*np.sqrt(n))
    # used in sampling
    #randomization_sampling = norm(loc=0, scale=0.5*np.sqrt(n))
    randomization_sampling = laplace(loc=0, scale=0.5*np.sqrt(n))
    
    E, s_E, beta, alpha_E, u_nE = solve(Z,D,Y,lagrange,randomizer,epsilon)
    
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(E)[0]) and E.sum() < p:

        # right now it is just the true covariance
        Sigmab0 = Sigma

        sampler = MH_sampler(Z, D, Y,
                         lagrange, randomizer, epsilon,
                         E, s_E, 
                         beta, alpha_E, u_nE, 
                         beta_star, Sigmab0,
                         randomization_sampling,
                         data_stepsize=data_step,
                         coefs_stepsize=coefs_step)
    
        samples = sampler.sample(ndraw, burnin) 
        observed = sampler.T_obs
              
        pval = np.sum(np.array(samples) > observed) * 1. / ndraw
        pval = 2 * min(pval, 1-pval)

        return pval

# compute the pvalue when H0: b0 = beta_star
# true_model means known covariance
def pvalue(Z, D, Y,
         lagrange, randomizer, epsilon,
         E, s_E,
         beta, alpha_E, u_nE, 
         beta_star,
         randomization_sampling,
         true_model=True,
         Sigma = None,
         ndraw=10000, burnin=10000, data_step=6., coefs_step=0.001):
    if true_model:
        Sigmab0 = Sigma
    else:
        raise ValueError('need to implement for unknown covariance!')
    sampler = MH_sampler(Z, D, Y,
                         lagrange, randomizer, epsilon,
                         E, s_E, 
                         beta, alpha_E, u_nE,
                         beta_star, Sigmab0,
                         randomization_sampling,
                         data_stepsize=data_step,
                         coefs_stepsize=coefs_step)
    samples = sampler.sample(ndraw, burnin)
    observed = sampler.T_obs
    pval = np.sum(np.array(samples) > observed) * 1. / ndraw
    pval = 2 * min(pval, 1-pval)
    return pval


        
