import time
import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov

def myround(a, decimals=1):
    a_x = np.round(a, decimals=1)* 10.
    rem = np.zeros(a.shape[0], bool)
    rem[(np.remainder(a_x, 2) == 1)] = 1
    a_x[rem] = a_x[rem] + 1.
    return a_x/10.


class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg)/(self.randomization_scale**2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg)/(self.randomization_scale**2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg>0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)
        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                        cube_prob[indicator]))/self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index])/
                                     ((prod_arg[pos_index]/arg_u[pos_index])-
                                      (1./arg_l[pos_index])))/self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] -(arg_l[neg_index]*neg_prod_arg[neg_index]))
                                    /self.randomization_scale)/(1.- neg_prod_arg[neg_index])


        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class approximate_conditional_prob_E(rr.smooth_atom):

    def __init__(self,
                 t, #point at which density is to computed
                 approx_density,
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t
        self.AD = approx_density
        self.q = self.AD.p - self.AD.nactive
        self.inactive_conjugate = self.active_conjugate = approx_density.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        lagrange = []
        for key, value in self.AD.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)

        self.inactive_lagrange = lagrange[~self.AD._overall]
        self.active_lagrange = lagrange[self.AD._overall]

        rr.smooth_atom.__init__(self,
                                (self.AD.nactive,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.AD.feasible_point,
                                coef=coef)

        self.coefs[:] = self.AD.feasible_point
        self.B_active = self.AD.opt_linear_term[:self.AD.nactive, :self.AD.nactive]
        self.B_inactive = self.AD.opt_linear_term[self.AD.nactive:, :self.AD.nactive]

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.AD.nactive)


    def sel_prob_smooth_objective(self, param, j, mode='both', check_feasibility=False):

        param = self.apply_offset(param)
        index = np.zeros(self.AD.nactive, bool)
        index[j] = 1
        data = np.squeeze(self.t * self.AD.target_linear_term[:, index]) \
               + self.AD.target_linear_term[:, ~index].dot(self.AD.target_observed[~index])

        offset_active = self.AD.opt_affine_term[:self.AD.nactive] + self.AD.null_statistic[:self.AD.nactive] + data[:self.AD.nactive]

        offset_inactive = self.AD.null_statistic[self.AD.nactive:] + data[self.AD.nactive:]

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.B_active, offset_active))

        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = 1.)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.B_inactive, offset_inactive))

        total_loss = rr.smooth_sum([active_conj_loss,
                                    cube_loss,
                                    self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, j, step=1, nstep=30, tol=1.e-6):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.sel_prob_smooth_objective(u, j, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, j, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current)

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                #print("current proposal and grad", proposal, newton_step)
                if np.all(proposal > 0):
                    break
                step *= 0.5
                if count >= 40:
                    #print(proposal)
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        # print('iter', itercount)
        value = objective(current)

        return current, value

class approximate_conditional_density_E(rr.smooth_atom, M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        M_estimator.__init__(self, loss, epsilon, penalty, randomization)

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

    def solve_approx(self):

        self.Msolve()
        self.feasible_point = np.abs(self.initial_soln[self._overall])
        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        nactive = self._overall.sum()

        Sigma_D_T = score_cov[:, :nactive]
        Sigma_T = score_cov[:nactive, :nactive]
        Sigma_T_inv = np.linalg.inv(Sigma_T)

        score_linear_term = self.score_transform[0]
        (self.opt_linear_term, self.opt_affine_term) = self.opt_transform

        # decomposition
        #print(self.opt_affine_term[nactive:])
        target_linear_term = (score_linear_term.dot(Sigma_D_T)).dot(Sigma_T_inv)

        # observed target and null statistic
        target_observed = self.observed_score_state[:nactive]
        null_statistic = (score_linear_term.dot(self.observed_score_state))-(target_linear_term.dot(target_observed))

        (self.target_linear_term, self.target_observed, self.null_statistic) \
            = (target_linear_term, target_observed, null_statistic)
        self.nactive = nactive

        #defining the grid on which marginal conditional densities will be evaluated
        grid_length = 120
        self.grid = np.linspace(-4, 8, num=grid_length)
        #s_obs = np.round(self.target_observed, decimals =1)

        print("observed values", target_observed)
        self.ind_obs = np.zeros(nactive, int)
        self.norm = np.zeros(nactive)
        self.h_approx = np.zeros((nactive, self.grid.shape[0]))

        for j in range(nactive):
            obs = target_observed[j]
            self.norm[j] = Sigma_T[j,j]
            if obs < self.grid[0]:
                self.ind_obs[j] = 0
            elif obs > np.max(self.grid):
                self.ind_obs[j] = grid_length
            else:
                self.ind_obs[j] = np.argmin(np.abs(self.grid-obs))

                #self.ind_obs[j] = (np.where(self.grid == obs)[0])[0]
            self.h_approx[j, :] = self.approx_conditional_prob(j)


    def approx_conditional_prob(self, j):
        h_hat = []

        for i in range(self.grid.shape[0]):

            approx = approximate_conditional_prob_E(self.grid[i], self)
            h_hat.append(-(approx.minimize2(j, nstep=50)[::-1])[0])

        return np.array(h_hat)


    def area_normalized_density(self, j, mean):

        normalizer = 0.

        approx_nonnormalized = []
        for i in range(self.grid.shape[0]):
            approx_density = np.exp(-np.true_divide((self.grid[i] - mean) ** 2, 2 * self.norm[j])
                                    + (self.h_approx[j,:])[i])

            normalizer += approx_density

            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer))

    def approximate_ci(self, j):

        param_grid = np.round(np.linspace(-5, 10, num=151), decimals=1)

        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):

            area_vec = self.area_normalized_density(j, param_grid[k])
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]

        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0



def test_approximate_ci_E(n=200, p=10, s=5, snr=5, rho=0.1,
                          lam_frac=1.,
                          loss='gaussian'):

    from selection.tests.instance import logistic_instance, gaussian_instance
    from selection.randomized.api import randomization

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    # randomizer = randomization.isotropic_gaussian((p,), scale=sigma)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    # W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), 1.)
    ci = approximate_conditional_density_E(loss, epsilon, penalty, randomization)

    ci.solve_approx()
    print("nactive", ci._overall.sum())
    active_set = np.asarray([i for i in range(p) if ci._overall[i]])

    true_support = np.asarray([i for i in range(p) if i < s])

    nactive = ci.nactive

    print("active set, true_support", active_set, true_support)

    #truth = np.round((np.linalg.pinv(X_1[:, active])).dot(X_1[:, active].dot(true_beta[active])))
    truth = beta[ci._overall]

    print("true coefficients", truth)

    if (set(active_set).intersection(set(true_support)) == set(true_support))== True:

        ci_active_E = np.zeros((nactive, 2))
        toc = time.time()
        for j in range(nactive):
            ci_active_E[j, :] = np.array(ci.approximate_ci(j))
            print(ci_active_E[j, :])
        tic = time.time()
        print('ci time now', tic - toc)
        #print('ci intervals now', ci_active_E)

        return active_set, ci_active_E, truth, nactive

    else:
        return 0

#test_approximate_ci_E()

def compute_coverage(p=10):

    niter = 50
    coverage = np.zeros(p)
    nsel = np.zeros(p)
    nerr = 0
    for iter in range(niter):
        print("\n")
        print("iteration", iter)
        try:
            test_ci = test_approximate_ci_E()
            if test_ci != 0:
                ci_active = test_ci[1]
                print("ci", ci_active)
                active_set = test_ci[0]
                true_val = test_ci[2]
                nactive = test_ci[3]
                toc = time.time()
                for l in range(nactive):
                    nsel[active_set[l]] += 1
                    print(true_val[l])
                    if (ci_active[l,0]<= true_val[l]) and (true_val[l]<= ci_active[l,1]):
                        coverage[active_set[l]] += 1
                tic = time.time()
                print('ci time', tic - toc)

            print(coverage[~np.isnan(coverage)])
            print(nsel[~np.isnan(nsel)])
            print('coverage so far',np.true_divide(np.sum(coverage[~np.isnan(coverage)]), np.sum(nsel[~np.isnan(nsel)])))

        except ValueError:
            nerr +=1
            print('ignore iteration raising ValueError')
            continue

    coverage_prop = np.true_divide(coverage, nsel)
    coverage_prop[coverage_prop == np.inf] = 0
    coverage_prop = np.nan_to_num(coverage_prop)
    return coverage_prop, nsel, nerr


print(compute_coverage())









