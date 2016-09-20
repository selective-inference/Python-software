import numpy as np
import regreg.api as rr
#import selection.sampling.randomized.api as randomized

from regreg.atoms.seminorms import seminorm
#from choldate import cholupdate, choldowndate

from scipy.stats import laplace, probplot, uniform
from selection.algorithms.lasso import instance

#get the selective_loss function used in gaussian_Xfixed
class selective_loss(rr.smooth_atom):

    ### begin API

    ### selective loss API

    def gradient(self, data, beta):
        """
        Gradient of smooth part.
        """
        raise NotImplementedError("abstract method")

    def hessian(self, data, beta):
        """
        Hessian of smooth part.
        """
        raise NotImplementedError("abstract method")

    def log_jacobian(self, data, beta):
        """
        Log-Jacobian of smooth part.
        Active subspace should have columns that
        """
        raise NotImplementedError("abstract method")

    def setup_sampling(self,
                       y,
                       quadratic_coef,
                       *args):
        raise NotImplementedError("abstract method")

    def proposal(self, data):
        """
        Metropolis-Hastings proposal to move `data`.
        """
        raise NotImplementedError("abstract method")

    def logpdf(self, y):
        """
        logpdf of `data`, refers to density `f` in the manuscript.
        """
        raise NotImplementedError("abstract method")

    def update_proposal(self, y, proposal, log_ratio):
        """
        Update state of loss based on current data,
        proposal and the acceptance probability of the step
        from y to proposal.
        """
        raise NotImplementedError("abstract method")

    def step_data(self, state, logpdf):

        self.total_data += 1

        data, opt_vars = state

        proposal, log_transition_ratio = self.proposal(data)

        #return proposal

        proposal_state = (proposal, opt_vars)

        log_ratio = (log_transition_ratio
                     + logpdf(proposal_state)
                     - logpdf(state))

        self.update_proposal(data, proposal, log_ratio)

        if np.log(np.random.uniform()) < log_ratio:
            self.accept_data += 1
            data = proposal

        return data

#getting the function gaussian_Xfixed used for the loss
class gaussian_Xfixed(selective_loss):
    def __init__(self, X, y,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):

        selective_loss.__init__(self, X.shape[1],
                                coef=coef,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.X = X
        self.y = y.copy()

    def smooth_objective(self, beta, mode='both',
                         check_feasibility=False):

        resid = self.y - np.dot(self.X, beta)

        if mode == 'both':
            f = self.scale((resid ** 2).sum()) / 2.
            g = self.scale(-np.dot(self.X.T, resid))
            return f, g
        elif mode == 'func':
            f = self.scale(np.linalg.norm(resid) ** 2) / 2.
            return f
        elif mode == 'grad':
            g = self.scale(-np.dot(self.X.T, resid))
            return g
        else:
            raise ValueError("mode incorrectly specified")

    # this is something that regreg does not know about, i.e.
    # what is data and what is not...

    def gradient(self, data, beta):
        """
        Gradient of smooth part restricted to active set
        """
        old_data, self.y = self.y, data
        g = self.smooth_objective(beta, 'grad')
        self.y = old_data
        return g

    def hessian(self):
        if not hasattr(self, "_XTX"):
            self._XTX = np.dot(self.X.T, self.X)
        return self._XTX

    def setup_sampling(self, y, mean, sigma, linear_part, value):

        ### JT: if sigma is known the variance should be adjusted
        ### if it is unknown then the pdf below should be uniform
        ### supported on sphere of some radius

        ### This can be implemented as part of
        ### a subclass

        self.accept_data = 0
        self.total_data = 0

        self.sigma = sigma

        P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
        I = np.identity(linear_part.shape[1])

        self.data = y
        self.mean = mean

        self.R = I - P

        self.P = P
        self.linear_part = linear_part

#getting function selective_penalty used in selective_l1norm_lan
class selective_penalty(seminorm):

    def setup_sampling(self,
                       gradient,
                       soln,
                       linear_randomization,
                       quadratic_coef):

        """
        Should store quadratic_coef.
        Its return value is the chosen parametrization
        of the selection event.
        In other API methods, this return value is
        referred to as `opt_vars`
        """

        raise NotImplementedError("abstract method")

    def form_subgradient(self, opt_vars):
        """
        Given the chosen parametrization
        of the selection event, this should form
        `z`, an element the subgradient of the penalty
        at `beta`.
        """
        raise NotImplementedError("abstract method")

    def form_parameters(self, opt_vars):
        """
        Given the chosen parametrization
        of the selection event, this should form
        `beta`.
        """
        raise NotImplementedError("abstract method")

    def form_optimization_vector(self, opt_vars):
        """
        Given the chosen parametrization
        of the selection event, this should form
        `(beta, z, epsilon * beta + z)`.
        """
        raise NotImplementedError("abstract method")

    def log_jacobian(self, hessian):
        """
        Given the Hessian of the loss at `beta`,
        compute the appropriate Jacobian which is the
        determinant of this matrix plus the Jacobian
        of the map $\epsilon \beta + z$
        """
        raise NotImplementedError("abstract method")

    def step_variables(self, state, randomization, logpdf, gradient):
        """
        State is a tuple (data, opt_vars).
        This method should take a Metropolis-Hastings
        step for `opt_vars`.
        The logpdf, is the callable that computes
        the density of the randomization,
        as well as the jacobian of the parameterization.
        randomization should be a callable that samples
        from the original randomization density.
        """
        raise NotImplementedError("abstract method")

#getting the penalty function used
class selective_l1norm_lan(rr.l1norm, selective_penalty):

    ### begin selective_penalty API

    ### API begins here

    def setup_sampling(self,
                       gradient,
                       soln,
                       linear_randomization,
                       quadratic_coef):

        # this will get used to randomize on simplex

        #self.simplex_randomization = (0.05, dirichlet(np.ones(self.shape)))
        self.quadratic_coef = quadratic_coef

        self.accept_l1_part, self.total_l1_part = 0, 0

        random_direction = quadratic_coef * soln + linear_randomization
        negative_subgrad = gradient + random_direction

        self.active_set = (soln != 0)

        self.signs = np.sign(soln[self.active_set])

        #abs_l1part = np.fabs(soln[self.active_set])
        #l1norm_ = abs_l1part.sum()

        subgrad = -negative_subgrad[self.inactive_set] # u_{-E}
        supnorm_ = np.fabs(negative_subgrad).max()

        if self.lagrange is None:
            raise NotImplementedError("only lagrange form is implemented")

        ##TODO: replace supnorm_ with self.lagrange? check whether they are the same
        ## it seems like supnorm_ is slightly bigger than self.lagrange

        betaE, cube = soln[self.active_set], subgrad / supnorm_

        #simplex, cube = np.fabs(soln[self.active_set]), subgrad / self.lagrange

        # print cube
        # for adaptive mcmc

        nactive = soln[self.active_set].shape[0]

        #self.chol_adapt = np.identity(nactive) / np.sqrt(nactive)

        return betaE, cube

    def form_subgradient(self, opt_vars):
        """
        opt_vars will be of the form returned by self.setup_sampling
        this should form z, the subgradient of P at beta
        """
        _, cube = opt_vars
        lam = self.lagrange

        active_set = self.active_set
        inactive_set = self.inactive_set

        full_subgrad = np.ones(self.shape)
        full_subgrad.flat[active_set] = self.signs
        full_subgrad.flat[inactive_set] = cube

        return lam * full_subgrad

    def form_parameters(self, opt_vars):
        """
        opt_vars will be of the form returned by self.setup_sampling
        this should form beta
        """

        betaE, _ = opt_vars

        full_params = np.zeros(self.shape)
        full_params.flat[self.active_set] = betaE

        return full_params

    def form_optimization_vector(self, opt_vars):
        """
        opt_vars will be of the form returned by self.setup_sampling
        this should form beta, z, epsilon * beta + z
        """


        # could be more efficient for LASSO

        params = self.form_parameters(opt_vars)
        subgrad = self.form_subgradient(opt_vars)
        return params, subgrad, self.quadratic_coef * params + subgrad

    def log_jacobian(self, hessian):
        if self.constant_log_jacobian is None:
            restricted_hessian = hessian[self.active_set][:,self.active_set]
            restricted_hessian += self.quadratic_coef * np.identity(restricted_hessian.shape[0])
            return np.linalg.slogdet(restricted_hessian)[1]
        else:
            return self.constant_log_jacobian

    #def step_variables(self, state, randomization, logpdf, gradient):
    #    """
    #    Updates internal parameterization of
    #    the optimization variables.

    #    """
    #    new_cube = self.step_cube(state, randomization, gradient)

    #    data, opt_vars = state
    #    simplex, _ = opt_vars
    #    new_state = (data, (simplex, new_cube))

    #    new_simplex = self.step_simplex(new_state, randomization, logpdf, gradient, self.hessian)

    #    return new_simplex, new_cube

    ### end selective_penalty API

    ### specific to lasso

    # for asymptotically Gaussian inference
    # the LASSO's Jacobian is (asymptotically) constant

    def get_constant_log_jacobian(self):
        if hasattr(self, "_constant_log_jacobian"):
            return self._constant_log_jacobian

    def set_constant_log_jacobian(self, log_jacobian):
        self._constant_log_jacobian = log_jacobian

    constant_log_jacobian = property(get_constant_log_jacobian,
                                     set_constant_log_jacobian)

    def get_active_set(self):
        if hasattr(self, "_active_set"):
            return self._active_set

    def set_active_set(self, active_set):
        self._active_set = np.zeros(self.shape, np.bool)
        self._active_set[active_set] = 1
        self._inactive_set = ~self._active_set

        nactive = self._active_set.sum()
        ninactive = self._inactive_set.sum()

        self.dtype = np.dtype([('betaE', (np.float,    # parameters
                                            nactive-1)), # on face simplex
                               ('scale', np.float),
                               ('signs', (np.int, nactive)),
                               ('cube', (np.float,
                                         ninactive))])

    active_set = property(get_active_set, set_active_set,
                          doc="The active set for selective sampling.")

    @property
    def inactive_set(self):
        if hasattr(self, "_inactive_set"):
            return self._inactive_set

    def get_penalty_params(self, scale):
        if self.lagrange is not None:
            return self.lagrange, scale
        else:
            return scale, self.bound




    def step_variables(self, state, randomization, logpdf, gradient, X, P, R):
        """
        Uses projected Langevin proposal ( X_{k+1} = P(X_k+\eta\grad\log\pi+\sqrt{2\pi} Z), where Z\sim\mathcal{N}(0,Id))
        for a new simplex point (a point in non-negative orthant, hence projection onto [0,\inf)^|E|)
        """
        self.total_l1_part += 1
        lam = self.lagrange

        data, opt_vars = state
        betaE, cube = opt_vars
        active = self.active_set
        #print 'active', active
        inactive = ~active
        #print 'inactive', inactive
        hessian = self.hessian

        if self.lagrange is None:
            raise NotImplementedError("The bound form has not been implemented")

        nactive = betaE.shape[0]
        ninactive = cube.shape[0]
        #stepsize = 1/np.sqrt(nactive)
        stepsize = 1/float(nactive+ninactive)  # eta below

        #stepsize = 0.1/float(nactive)
        # new for projected Langevin MCMC

        _ , _ , opt_vec = self.form_optimization_vector(opt_vars) # opt_vec=\epsilon(\beta 0)+u, u=\grad P(\beta), P penalty

        sign_vec =  - np.sign(gradient + opt_vec)  # sign(w), w=grad+\epsilon*beta+lambda*u

        #restricted_hessian = hessian[self.active_set][:, active]
        B = hessian+self.quadratic_coef*np.identity(nactive+ninactive)
        A = B[:, active]
        #A1 = hessian[active][:, active] + self.quadratic_coef*np.identity(nactive)
        #A2 = hessian[inactive][:, active]

        #A=np.concatenate((A1, A2), axis=0)
        # the following is \grad_{\beta}\log g(w), w = \grad l(\beta)+\epsilon (\beta 0)+\lambda u = A*\beta+b,
        # becomes - \grad_{\beta}\|w\|_1 = - \grad_{\beta}\|A*\beta+b\|_1=A^T*sign(A*\beta+b)
        # A = hessian+\epsilon*Id (symmetric), A*\beta+b = gradient+opt_vec
        # \grad\log\pi if we want a sample from a distribution \pi

        grad_log_pi =  np.dot(A.T, sign_vec)

        # proposal = Proj(simplex+\eta*grad_{\beta}\log g+\sqrt{2\eta}*Z), Z\sim\mathcal{N}(0, Id)
        # projection on the non-negative orthant
        # print np.sum(simplex+(stepsize*grad_log_pi)+(np.sqrt(2*stepsize)*np.random.standard_normal(nactive))<0)
        #proposal = np.clip(simplex+(stepsize*grad_log_pi)+(np.sqrt(2*stepsize)*np.random.standard_normal(nactive)), 0, np.inf)

        betaE_proposal = betaE+(stepsize*grad_log_pi)+(np.sqrt(2*stepsize)*np.random.standard_normal(nactive))

        for i in range(nactive):
            if (betaE_proposal[i]*self.signs[i]<0):
                    betaE_proposal[i] = 0


        grad_cube_log_pi =  self.lagrange*sign_vec[inactive]
        cube_proposal = cube + (stepsize*grad_cube_log_pi)+(np.sqrt(2*stepsize)*np.random.standard_normal(ninactive))
        cube_proposal = np.clip(cube_proposal, -1, 1)


        grad_y_log_pi = - (data + np.dot(X, sign_vec))
        data_proposal = data + (stepsize*grad_y_log_pi)+(np.sqrt(2*stepsize)*np.random.standard_normal(data.shape[0]))


        data_proposal = np.dot(P, data) + np.dot(R, data_proposal)

        #rand = randomization
        #random_sample = rand.rvs(size=nactive)
        #step = np.dot(self.chol_adapt, random_sample)
        #print np.sum(simplex+step<0)
        #proposal = np.fabs(simplex + step)

        #log_ratio = (logpdf((data_proposal, (betaE_proposal, cube_proposal)))-logpdf(state))
        #log_ratio = log_ratio+((-np.dot(data_proposal, data_proposal) + np.dot(data,data))/float(2))

         # update cholesky factor

        #alpha = np.minimum(np.exp(log_ratio), 1)
        #target = 2.4 / np.sqrt(nactive)
        #multiplier = ((self.total_l1_part+1)**(-0.8) *
        #               (np.exp(log_ratio) - target))
        #rank_one = np.sqrt(np.fabs(multiplier)) * step / np.linalg.norm(random_sample)

        #if multiplier > 0:
        #     cholupdate(self.chol_adapt, rank_one) # update done in place
        #else:
        #     choldowndate(self.chol_adapt, rank_one) # update done in place

        # return proposal

        #if np.log(np.random.uniform()) < log_ratio:
        data, betaE, cube = data_proposal, betaE_proposal, cube_proposal
        opt_vars = (betaE, cube)
        self.accept_l1_part += 1

        return data, opt_vars

def selection(X, y, random_Z, randomization_scale=1, sigma=1):
    n, p = X.shape
    loss = gaussian_Xfixed(X, y)
    epsilon = 1. / np.sqrt(n)
    # epsilon = 1.
    lam_frac = 1.
    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    penalty = selective_l1norm_lan(p, lagrange=lam)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, -randomization_scale * random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}


    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    active = (initial_soln != 0)
    if np.sum(active) == 0:
        return -1, -1, np.nan, np.nan, np.nan, np.nan
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE, cube = penalty.setup_sampling(initial_grad,
                                         initial_soln,
                                         -random_Z,
                                         epsilon)
    #active = penalty.active_set
    subgradient = -(initial_grad+epsilon*initial_soln-randomization_scale*random_Z)
    cube = subgradient[~active]/lam
    return lam, epsilon, active, betaE, cube, initial_soln

#creating instance X,y,beta

class instance(object):

    def __init__(self, n, p, s, snr=5, sigma=1., rho=0, random_signs=True, scale =True, center=True):
         (self.n, self.p, self.s,
         self.snr,
         self.sigma,
         self.rho) = (n, p, s,
                     snr,
                     sigma,
                     rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
              np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         self.beta = np.zeros(p)
         self.beta[:self.s] = self.snr
         if random_signs:
             self.beta[:self.s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
         self.active = np.zeros(p, np.bool)
         self.active[:self.s] = True

    def _noise(self):
        return np.random.standard_normal(self.n)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + self._noise()) * self.sigma
        return self.X, Y, self.beta * self.sigma, np.nonzero(self.active)[0], self.sigma

####check if code is working
n=100
p=20
s=5
snr=5
data_instance = instance(n, p, s, snr)
X, y, true_beta, nonzero, sigma = data_instance.generate_response()
#print true_beta
random_Z = np.random.standard_normal(p)
lam, epsilon, active, betaE, cube, initial_soln = selection(X, y, random_Z)