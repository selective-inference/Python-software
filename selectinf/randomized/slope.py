from __future__ import print_function

import functools
import numpy as np

# sklearn imports

have_isotonic = False
try:
    from sklearn.isotonic import IsotonicRegression
    have_isotonic = True
except ImportError:
    raise ValueError('unable to import isotonic regression from sklearn, SLOPE subgradient projection will not work')

# regreg imports

from regreg.atoms.slope import _basic_proximal_map
import regreg.api as rr

from ..constraints.affine import constraints

from .randomization import randomization
from ..base import restricted_estimator, _compute_hessian
from .query import gaussian_query
from .lasso import lasso

class slope(gaussian_query):

    def __init__(self,
                 loglike,
                 slope_weights,
                 ridge_term,
                 randomizer,
                 perturb=None):
        r"""
        Create a new post-selection object for the SLOPE problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        slope_weights : np.ndarray
            SLOPE weights for L-1 penalty. If a float,
            it is broadcast to all features.

        ridge_term : float
            How big a ridge term to add?

        randomizer : object
            Randomizer -- contains representation of randomization density.

        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(slope_weights).shape == ():
            slope_weights = np.ones(loglike.shape) * slope_weights
        self.slope_weights = np.asarray(slope_weights)

        self.randomizer = randomizer 
        self.ridge_term = ridge_term
        self.penalty = rr.slope(slope_weights, lagrange=1.)
        self._initial_omega = perturb  # random perturbation

    def _solve_randomized_problem(self, 
                                  perturb=None, 
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):
        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        observed_soln = problem.solve(quad, **solve_args)
        observed_subgrad = -(self.loglike.smooth_objective(observed_soln, 'grad') +
                            quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        self.observed_soln, self.observed_subgrad = self._solve_randomized_problem(perturb=perturb, solve_args=solve_args)
        p = self.observed_soln.shape[0]

        # now we have to work out SLOPE details, clusters, etc.

        active_signs = np.sign(self.observed_soln)
        active = self._active = active_signs != 0

        self._overall = overall = active> 0
        self._inactive = inactive = ~self._overall

        _active_signs = active_signs.copy()
        self.selection_variable = {'sign': _active_signs,
                                   'variables': np.nonzero(self._overall)[0]}


        indices = self.selection_variable['indices'] = np.argsort(-np.fabs(self.observed_soln))
        sorted_soln = self.observed_soln[indices]
        initial_scalings = np.sort(np.unique(np.fabs(self.observed_soln[active])))[::-1]
        self.observed_opt_state = initial_scalings
        self._unpenalized = np.zeros(p, np.bool)

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        self.num_opt_var = self.observed_opt_state.shape[0]

        self._unscaled_cov_score, _hessian_active = _compute_hessian(self.loglike,
                                                                     beta_bar,
                                                                     active)


        _score_linear_term = -_hessian_active
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        cur_indx_array = []
        cur_indx_array.append(0)
        cur_indx = 0
        pointer = 0
        signs_cluster = []
        for j in range(p - 1):
            if np.abs(sorted_soln[j + 1]) != np.abs(sorted_soln[cur_indx]):
                cur_indx_array.append(j + 1)
                cur_indx = j + 1
                sign_vec = np.zeros(p)
                sign_vec[np.arange(j + 1 - cur_indx_array[pointer]) + cur_indx_array[pointer]] = \
                    np.sign(self.observed_soln[indices[np.arange(j + 1 - cur_indx_array[pointer]) + cur_indx_array[pointer]]])
                signs_cluster.append(sign_vec)
                pointer = pointer + 1
                if sorted_soln[j + 1] == 0:
                    break

        signs_cluster = np.asarray(signs_cluster).T
        self.selection_variable['signs_cluster'] = signs_cluster

        if signs_cluster.size == 0:
            return active_signs
        else:
            X, y = self.loglike.data
            X_clustered = X[:, indices].dot(signs_cluster)
            _opt_linear_term = X.T.dot(X_clustered)

            # now make the constraints

            self._setup = True
            A_scaling_0 = -np.identity(self.num_opt_var)
            A_scaling_1 = -np.identity(self.num_opt_var)[:(self.num_opt_var - 1), :]
            for k in range(A_scaling_1.shape[0]):
                A_scaling_1[k, k + 1] = 1
            A_scaling = np.vstack([A_scaling_0, A_scaling_1])
            b_scaling = np.zeros(2 * self.num_opt_var - 1)

            self._setup_sampler_data = (A_scaling,
                                        b_scaling,
                                        _opt_linear_term,
                                        self.observed_subgrad)
            self.opt_linear = _opt_linear_term
            return active_signs

    def setup_inference(self,
                        dispersion):

        if self.num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data,
                                dispersion=dispersion)


    # Targets of inference
    # and covariance with score representation
    # are same as LASSO

    @staticmethod
    def gaussian(X,
                 Y,
                 slope_weights,
                 sigma=1.,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None):

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return slope(loglike, 
                     np.asarray(slope_weights) / sigma ** 2, 
                     ridge_term, 
                     randomizer)

# split SLOPE

class split_slope(lasso):

    """
    Data split, then LASSO (i.e. data carving)
    """

    def __init__(self,
                 loglike,
                 slope_weights,
                 proportion_select,
                 ridge_term=0,
                 perturb=None,
                 estimate_dispersion=True):

        (self.loglike,
         self.slope_weights,
         self.proportion_select,
         self.ridge_term) = (loglike,
                             slope_weights,
                             proportion_select,
                             ridge_term)

        self.nfeature = p = self.loglike.shape[0]
        self.penalty = rr.slope(slope_weights, lagrange=1.)
        self._initial_omega = perturb  # random perturbation
        self.estimate_dispersion = estimate_dispersion

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        signs = slope.fit(self, 
                          solve_args=solve_args,
                          perturb=perturb)
        
        # for data splitting randomization,
        # we need to estimate a dispersion parameter

        # we then setup up the sampler again
        df_fit = len(self.selection_variable['variables'])

        if self.estimate_dispersion:

            X, y = self.loglike.data
            n, p = X.shape

            dispersion = 2 * (self.loglike.smooth_objective(self._beta_full,
                                                            'func') /
                              (n - df_fit))

            self.dispersion_ = dispersion
            # run setup again after
            # estimating dispersion

        self.df_fit = df_fit

        return signs


    def setup_inference(self,
                        dispersion):

        if self.df_fit > 0:

            if dispersion is None:
                self._setup_sampler(*self._setup_sampler_data,
                                    dispersion=self.dispersion_)

            else:
                self._setup_sampler(*self._setup_sampler_data,
                                    dispersion=dispersion)

    def _setup_implied_gaussian(self, 
                                opt_linear, 
                                observed_subgrad,
                                dispersion=1):

        # key observation is that the covariance of the added noise is 
        # roughly dispersion * (1 - pi) / pi * X^TX (in OLS regression, similar for other
        # models), so the precision is  (X^TX)^{-1} * (pi / ((1 - pi) * dispersion))
        # and prec.dot(opt_linear) = S_E / (dispersion * (1 - pi) / pi)
        # because opt_linear has shape p x E with the columns
        # being those non-zero columns of the solution. Above S_E = np.diag(signs)
        # the conditional precision is S_E Q[E][:,E] * pi / ((1 - pi) * dispersion) S_E
        # and regress_opt is -Q[E][:,E]^{-1} S_E
        # padded with zeros
        # to be E x p

        pi_s = self.proportion_select
        ratio = (1 - pi_s) / pi_s

        ordered_vars = self.selection_variable['variables']
        indices = self.selection_variable['indices']
        signs_cluster = self.selection_variable['signs_cluster']
        
        # JT: this may be expensive to form -- not pxp but large
        cond_precision = signs_cluster.T.dot(self.opt_linear[indices] / (dispersion * ratio))

        assert(np.linalg.norm(cond_precision - cond_precision.T) / 
               np.linalg.norm(cond_precision) < 1.e-6)
        cond_cov = np.linalg.inv(cond_precision)
        regress_opt = np.zeros((len(ordered_vars),
                                   self.nfeature)) 
        # JT: not sure this is right -- had to remove signs
        regress_opt[:, ordered_vars] = -cond_cov / (dispersion * ratio)
        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        ## probably missing a dispersion in the denominator
        prod_score_prec_unnorm = np.identity(self.nfeature) / (dispersion * ratio)

        ## probably missing a multiplicative factor of ratio
        cov_rand = self._unscaled_cov_score * (dispersion * ratio)

        M1 = prod_score_prec_unnorm * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        M4 = M1.dot(opt_linear)
        M3 = M4.dot(cond_cov).dot(M4.T)
    
        # would be nice to not store these?
        
        self.M1 = M1  
        self.M2 = M2
        self.M3 = M3
        self.M4 = M4
        self.M5 = M1.dot(self.observed_score_state + observed_subgrad)

        return (cond_mean,
                cond_cov,
                cond_precision,
                M1,
                M2,
                M3,
                self.M4,
                self.M5)

    def _solve_randomized_problem(self, 
                                  # optional binary vector 
                                  # indicating selection data 
                                  perturb=None, 
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if none supplied
        if perturb is not None:
            self._selection_idx = perturb
        if not hasattr(self, "_selection_idx"):
            X, y = self.loglike.data
            total_size = n = X.shape[0]
            pi_s = self.proportion_select
            self._selection_idx = np.zeros(n, np.bool)
            self._selection_idx[:int(pi_s*n)] = True
            np.random.shuffle(self._selection_idx)

        inv_frac = 1 / self.proportion_select
        quad = rr.identity_quadratic(self.ridge_term,
                                     0,
                                     0,
                                     0)
        
        randomized_loss = self.loglike.subsample(self._selection_idx)
        randomized_loss.coef *= inv_frac

        problem = rr.simple_problem(randomized_loss, self.penalty)
        observed_soln = problem.solve(quad, **solve_args) 
        observed_subgrad = -(randomized_loss.smooth_objective(observed_soln,
                                                             'grad') +
                            quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad

    @staticmethod
    def gaussian(X,
                 Y,
                 slope_weights,
                 proportion,
                 sigma=1.,
                 quadratic=None,
                 estimate_dispersion=True):
        r"""
        Squared-error LASSO with feature weights.
        Objective function is (before randomization)

        .. math::

            \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + 
           \sum_{i=1}^p \lambda_i |\beta_i|

        where $\lambda$ is `slope_weights`. The ridge term
        is determined by the Hessian and `np.std(Y)` by default.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        Y : ndarray
            Shape (n,) -- the response.

        slope_weights: [float, sequence]

        proportion: float
            What proportion of data to use for selection.
 
        sigma : float (optional)
            Noise variance. Set to 1 if `covariance_estimator` is not None.
            This scales the loglikelihood by `sigma**(-2)`.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.slope.slope`

        """

        loglike = rr.glm.gaussian(X, 
                                  Y, 
                                  coef=1. / sigma ** 2, 
                                  quadratic=quadratic)

        return split_slope(loglike, 
                           np.asarray(slope_weights)/sigma**2,
                           proportion,
                           estimate_dispersion=estimate_dispersion)


    @staticmethod
    def logistic(X,
                 successes,
                 slope_weights,
                 proportion,
                 trials=None,
                 quadratic=None):
        r"""
        Logistic LASSO with feature weights (before randomization)

        .. math::

             \beta \mapsto \ell(X\beta) + \sum_{i=1}^p \lambda_i |\beta_i|

        where $\ell$ is the negative of the logistic
        log-likelihood (half the logistic deviance)
        and $\lambda$ is `slope_weights`.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        successes : ndarray
            Shape (n,) -- response vector. An integer number of successes.
            For data that is proportions, multiply the proportions
            by the number of trials first.

        slope_weights: [float, sequence]

        proportion: float
            What proportion of data to use for selection.
 
        trials : ndarray (optional)
            Number of trials per response, defaults to
            ones the same shape as Y.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.slope.slope`

        """

        loglike = rr.glm.logistic(X,
                                  successes,
                                  trials=trials,
                                  quadratic=quadratic)

        return split_slope(loglike, 
                           np.asarray(slope_weights),
                           proportion)

    @staticmethod
    def coxph(X,
              times,
              status,
              slope_weights,
              proportion,
              quadratic=None):
        r"""
        Cox proportional hazards LASSO with feature weights.
        Objective function is (before randomization)

        .. math::

            \beta \mapsto \ell^{\text{Cox}}(\beta) + 
            \sum_{i=1}^p \lambda_i |\beta_i|

        where $\ell^{\text{Cox}}$ is the
        negative of the log of the Cox partial
        likelihood and $\lambda$ is `slope_weights`.
        Uses Efron's tie breaking method.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        times : ndarray
            Shape (n,) -- the survival times.

        status : ndarray
            Shape (n,) -- the censoring status.

        slope_weights: [float, sequence]


        proportion: float
            What proportion of data to use for selection.
 
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.slope.slope`

        """
        n, p = X.shape
        loglike = rr.glm.cox(X, times, status, quadratic=quadratic)

        return split_slope(loglike, 
                           np.asarray(slope_weights),
                           proportion)

    @staticmethod
    def poisson(X,
                counts,
                slope_weights,
                proportion,
                quadratic=None,
                ridge_term=0):
        r"""
        Poisson log-linear LASSO with feature weights.
        Objective function is (before randomization)

        .. math::

            \beta \mapsto \ell^{\text{Poisson}}(\beta) + \sum_{i=1}^p \lambda_i |\beta_i|

        where $\ell^{\text{Poisson}}$ is the negative
        of the log of the Poisson likelihood (half the deviance)
        and $\lambda$ is `slope_weights`.

        Parameters
        ----------

        X : ndarray
            Shape (n,p) -- the design matrix.

        counts : ndarray
            Shape (n,) -- the response.

        slope_weights: [float, sequence]

        proportion: float
            What proportion of data to use for selection.
 
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.

        Returns
        -------

        L : `selection.randomized.slope.slope`

        """
        loglike = rr.glm.poisson(X, counts, quadratic=quadratic)

        return split_slope(loglike, 
                           np.asarray(slope_weights),
                           proportion)



# Projection onto selected subgradients of SLOPE

def _projection_onto_selected_subgradients(prox_arg,
                                           weights,
                                           ordering,
                                           cluster_sizes,
                                           active_signs,
                                           last_value_zero=True):
    """
    Compute the projection of a point onto the set of
    subgradients of the SLOPE penalty with a given
    clustering of the solution and signs of the variables.
    This is a projection onto a lower dimensional set. The dimension
    of this set is p -- the dimensions of the `prox_arg` minus
    the number of unique values in `ordered_clustering` + 1 if the
    last value of the solution was zero (i.e. solution was sparse).

    Parameters
    ----------

    prox_arg : np.ndarray(p, np.float)
        Point to project

    weights : np.ndarray(p, np.float)
        Weights of the SLOPE penalty.

    ordering : np.ndarray(p, np.int)
        Order of original argument to SLOPE prox.
        First entry corresponds to largest argument of SLOPE prox.

    cluster_sizes : sequence
        Sizes of clusters, starting with
        largest in absolute value.

    active_signs : np.ndarray(p, np.int)
         Signs of non-zero coefficients.

    last_value_zero : bool
        Is the last solution value equal to 0?
    """

    result = np.zeros_like(prox_arg)

    ordered_clustering = []
    cur_idx = 0
    for cluster_size in cluster_sizes:
        ordered_clustering.append([ordering[j + cur_idx] for j in range(cluster_size)])
        cur_idx += cluster_size

    # Now, run appropriate SLOPE prox on each cluster
    cur_idx = 0
    for i, cluster in enumerate(ordered_clustering):
        prox_subarg = np.array([prox_arg[j] for j in cluster])

        # If the value of the soln to the prox was non-zero
        # then we solve a SLOPE of size 1 smaller than the cluster

        # If the cluster size is 1, the value is just
        # the corresponding signed weight

        if i < len(ordered_clustering) - 1 or not last_value_zero:
            if len(cluster) == 1:
                result[cluster[0]] = weights[cur_idx] * active_signs[cluster[0]]
            else:
                indices = [j + cur_idx for j in range(len(cluster))]
                cluster_weights = weights[indices]

                ir = IsotonicRegression()
                _ir_result = ir.fit_transform(np.arange(len(cluster)), cluster_weights[::-1])[::-1]
                result[indices] = -np.multiply(active_signs[indices], _ir_result/2.)

        else:
            indices = np.array([j + cur_idx for j in range(len(cluster))])
            cluster_weights = weights[indices]

            slope_prox = _basic_proximal_map(prox_subarg, cluster_weights)
            result[indices] = prox_subarg - slope_prox

        cur_idx += len(cluster)

    return result
