import functools

import numpy as np
import regreg.api as rr
from ..constraints.affine import constraints

from .query import gaussian_query
from .randomization import randomization

class modelQ(gaussian_query):

    r"""
    A class for the randomized LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} -X^Ty + \frac{1}{2} \beta^TQ\beta + 
            \sum_{i=1}^p \lambda_i |\beta_i\| - \omega^T\beta + \frac{\epsilon}{2} \|\beta\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty. Each static method
    forms $\ell$ as well as the $\ell_1$ penalty. The generic class
    forms the remaining two terms in the objective.

    """

    def __init__(self, 
                 Q,
                 X, 
                 y,
                 feature_weights,
                 ridge_term=None,
                 randomizer_scale=None,
                 perturb=None):
        r"""

        Create a new post-selection object for the LASSO problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        feature_weights : np.ndarray
            Feature weights for L-1 penalty. If a float,
            it is brodcast to all features.

        ridge_term : float
            How big a ridge term to add?

        randomizer_scale : float
            Scale for IID components of randomization.

        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.

        """

        (self.Q,
         self.X,
         self.y) = (Q, X, y)

        self.loss = rr.quadratic_loss(Q.shape[0], Q=Q)
        n, p = X.shape
        self.nfeature = p

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        mean_diag = np.diag(Q).mean()
        if ridge_term is None:
            ridge_term = np.std(y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(y) * np.sqrt(n / (n - 1.))

        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        self.ridge_term = ridge_term
        self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)
        self._initial_omega = perturb # random perturbation

    def fit(self, 
            solve_args={'tol':1.e-12, 'min_its':50}, 
            perturb=None):
        """
        Fit the randomized lasso using `regreg`.

        Parameters
        ----------

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        Returns
        -------

        signs : np.float
             Support and non-zero signs of randomized lasso solution.
             
        """

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        quad_data = rr.identity_quadratic(0, 0, -self.X.T.dot(self.y), 0)
        problem = rr.simple_problem(self.loss, self.penalty)
        self.initial_soln = problem.solve(quad + quad_data, **solve_args)

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        self.selection_variable = {'sign':_active_signs,
                                   'variables':self._overall}

        # initial state for opt variables

        initial_subgrad = -(self.loss.smooth_objective(self.initial_soln, 'grad') + 
                            quad_data.objective(self.initial_soln, 'grad') +
                            quad.objective(self.initial_soln, 'grad')) 
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.fabs(self.initial_soln[active])
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized])

        E = overall
        Q_E = self.Q[E][:,E]
        _beta_unpenalized = np.linalg.inv(Q_E).dot(self.X[:,E].T.dot(self.y))
        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # observed state for score in internal coordinates

        self.observed_internal_state = np.hstack([_beta_unpenalized,
                                                  -self.loss.smooth_objective(beta_bar, 'grad')[inactive] + 
                                                  quad_data.objective(beta_bar, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self.num_opt_var))
        _score_linear_term = np.zeros((p, self.num_opt_var))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        X, y = self.X, self.y
        _hessian_active = self.Q[:, active]
        _hessian_unpen = self.Q[:, unpenalized]

        _score_linear_term = -np.hstack([_hessian_active, _hessian_unpen])

        # set the observed score (data dependent) state

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += (self.loss.smooth_objective(beta_bar, 'grad')[inactive] + 
                                                quad_data.objective(beta_bar, 'grad')[inactive])

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p, j, active_signs[j]) for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = _hessian_active * active_signs[None, active] + self.ridge_term * active_directions
        _opt_linear_term[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), self.num_opt_var)
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (_hessian_unpen
                                                      + self.ridge_term * unpenalized_directions) 

        # two transforms that encode score and optimization
        # variable roles 

        self.opt_transform = (_opt_linear_term, self.initial_subgrad)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self._setup = True
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.ndim = self.loss.shape[0]

        # compute implied mean and covariance

        opt_linear, opt_offset = self.opt_transform

        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        self._setup_sampler(A_scaling,
                            b_scaling,
                            opt_linear,
                            opt_offset)
        
        return active_signs

    def summary(self,
                target="selected",
                features=None,
                parameter=None,
                level=0.9,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=False,
                dispersion=None):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        target : one of ['selected', 'full']

        features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        compute_intervals : bool
            Compute confidence intervals?

        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?

        """

        if parameter is None:
            parameter = np.zeros(self.loss.shape[0])

        if target == 'selected':
            (observed_target, 
             cov_target, 
             cov_target_score, 
             alternatives) = self.selected_targets(features=features, 
                                                   dispersion=dispersion)
        else:
            X, y = self.loglike.data
            n, p = X.shape
            if n > p and target == 'full':
                (observed_target, 
                 cov_target, 
                 cov_target_score, 
                 alternatives) = self.full_targets(features=features, 
                                                   dispersion=dispersion)
            else:
                raise NotImplementedError
                (observed_target, 
                 cov_target, 
                 cov_target_score, 
                 alternatives) = self.debiased_targets(features=features, 
                                                       dispersion=dispersion)

        if self._overall.sum() > 0:
            opt_sample = self.sampler.sample(ndraw,  burnin)

            pivots = self.sampler.coefficient_pvalues(observed_target, 
                                                      cov_target, 
                                                      cov_target_score, 
                                                      parameter=parameter, 
                                                      sample=opt_sample, 
                                                      alternatives=alternatives)
            if not np.all(parameter == 0):
                pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                           cov_target, 
                                                           cov_target_score, 
                                                           parameter=np.zeros_like(parameter), 
                                                           sample=opt_sample, 
                                                           alternatives=alternatives)
            else:
                pvalues = pivots

            intervals = None
            if compute_intervals:
                intervals = self.sampler.confidence_intervals(observed_target, 
                                                              cov_target, 
                                                              cov_target_score,
                                                              sample=opt_sample)

            return pivots, pvalues, intervals
        else:
            return [], [], []


    def selective_MLE(self,
                      target="selected",
                      features=None,
                      parameter=None,
                      level=0.9,
                      compute_intervals=False,
                      dispersion=None,
                      solve_args={'tol':1.e-12}):
        """

        Parameters
        ----------

        target : one of ['selected', 'full']

        features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        compute_intervals : bool
            Compute confidence intervals?

        dispersion : float (optional)
            Use a known value for dispersion, or Pearson's X^2?

        """

        if parameter is None:
            parameter = np.zeros(self.loss.shape[0])

        observed_target, cov_target, cov_target_score, alternatives = self.selected_targets(features=features, dispersion=dispersion)

        # working out conditional law of opt variables given
        # target after decomposing score wrt target

        return self.sampler.selective_MLE(observed_target, 
                                          cov_target, 
                                          cov_target_score, 
                                          self.observed_opt_state,
                                          solve_args=solve_args)

    def selected_targets(self, features=None, dispersion=None):

        X, y = self.X, self.y
        n, p = X.shape

        if features is None:
            active = self._active
            unpenalized = self._unpenalized
            noverall = active.sum() + unpenalized.sum()
            overall = active + unpenalized

            Xfeat = X[:,overall]
            score_linear = self.score_transform[0]
            Q = -score_linear[overall]
            Qi = np.linalg.inv(Q)
            cov_target = Qi.dot(Xfeat.T.dot(Xfeat)).dot(Qi) # sandwich estimator
            observed_target = self._beta_full[overall]
            crosscov_target_score = score_linear.dot(cov_target)
            print(cov_target[:5][:,:5])
            alternatives = [{1:'greater', -1:'less'}[int(s)] for s in self.selection_variable['sign'][active]] + ['twosided'] * unpenalized.sum()

        else:

            features_b = np.zeros_like(self._overall)
            features_b[features] = True
            features = features_b

            Xfeat = X[:,features]
            Qfeat = self.Q[features][:,features]
            Gfeat = self.loss.smooth_objective(self.initial_soln, 'grad')[features] - Xfeat.T.dot(y)
            Qfeat_inv = np.linalg.inv(Qfeat)
            one_step = self.initial_soln[features] - Qfeat_inv.dot(Gfeat)
            cov_target = Qfeat_inv.dot(Xfeat.T.dot(Xfeat)).dot(Qfeat_inv)
            _score_linear = -self.Q[features]
            crosscov_target_score = _score_linear.dot(cov_target)
            observed_target = one_step
            alternatives = ['twosided'] * features.sum()

        if dispersion is None: # use Pearson's X^2
            relaxed = np.linalg.pinv(Xfeat).dot(y)
            dispersion = ((y - Xfeat.dot(relaxed))**2).sum() / (n - Xfeat.shape[1])
        print(dispersion, 'dispersion')

        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    def full_targets(self, features=None, dispersion=None):

        if features is None:
            features = self._overall
        features_bool = np.zeros(self._overall.shape, np.bool)
        features_bool[features] = True
        features = features_bool

        X, y = self.loglike.data
        n, p = X.shape

        # target is one-step estimator

        Qfull = self.Q
        G = self.loss.smooth_objective(self.initial_soln, 'grad') - X.T.dot(y)
        Qfull_inv = np.linalg.inv(Qfull)
        one_step = self.initial_soln - Qfull_inv.dot(G)
        cov_target = Qfull_inv[features][:,features]
        observed_target = one_step[features]
        crosscov_target_score = np.zeros((p, cov_target.shape[0]))
        crosscov_target_score[features] = -np.identity(cov_target.shape[0])

        if dispersion is None: # use Pearson's X^2
            dispersion = ((y - self.loglike.saturated_loss.mean_function(X.dot(one_step)))**2 / self._W).sum() / (n - p)

        alternatives = ['twosided'] * features.sum()
        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

