import numpy as np
import regreg.api as rr
from scipy.linalg import cho_solve, cho_factor

class softmax_conjugate(rr.smooth_atom):

    """

    Objective function that computes the value of 

    .. math..

        \inf_{\mu: A\mu \leq b} \frac{1}{2} \|y-z\|^2_2 + \sum_{i=1}^n \log(1 + 1 /(b_i - a_i^T\mu))

    """

    def __init__(self, 
                 affine_con, 
                 feasible_point,
                 sigma=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                                affine_con.linear_part.shape[1],
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.affine_con = affine_con
        self.feasible_point = feasible_point
        self.sigma = sigma

    def smooth_objective(self, natural_param, mode='func', check_feasibility=False):

        natural_param = self.apply_offset(natural_param)

        value, minimizer = self._solve_conjugate_problem(natural_param)

        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(minimizer)
        elif mode == 'both':
            return self.scale(value), self.scale(minimizer)
        else:
            raise ValueError('mode incorrectly specified')

    def _solve_conjugate_problem(self, natural_param, niter=500, tol=1.e-10):

        affine_con = self.affine_con

        loss = softmax(affine_con, sigma=self.sigma)

        L = rr.identity_quadratic(0, 0, -natural_param, 0) # linear_term
        A = affine_con.linear_part
        b = affine_con.offset
        mean_param = self.feasible_point.copy()
        step = 1. / self.sigma
        f_cur = np.inf
        for i in range(niter):
            G = -natural_param + loss.smooth_objective(mean_param, 'grad')
            proposed = mean_param - step * G
            slack = b - A.dot(proposed) 
            if i % 5 == 0:
                step *= 2.
            if np.any(slack < 0):
                step *= 0.5
            else:

                f_proposed = (-(natural_param * proposed).sum() +
                               loss.smooth_objective(proposed, 'func'))

                if f_proposed > f_cur * (1 + tol):
                    step *= 0.5
                else:
                    mean_param = proposed
                    if np.fabs(f_cur - f_proposed) < tol * max([1, 
                                                                np.fabs(f_cur), 
                                                                np.fabs(f_proposed)]):
                        break
                    f_cur = f_proposed

        return -f_proposed, mean_param

class softmax(rr.smooth_atom):

    """
    Softmax function

    .. math..

        \mu \mapsto \frac{1}{2\sigma^2} \|\mu\|^2 + 
        \sum_{i=1}^n \log(1 + \sigma /(b_i - a_i^T\mu))

    """

    def __init__(self, 
                 affine_con, 
                 sigma=1.,
                 feasible_point=None,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                                affine_con.linear_part.shape[1],
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.affine_con = affine_con
        self.feasible_point = feasible_point
        self.sigma = sigma

    def smooth_objective(self, 
                         mean_param, 
                         mode='func', 
                         check_feasibility=False):

        mean_param = self.apply_offset(mean_param)
        A = self.affine_con.linear_part
        b = self.affine_con.offset

        slack = b - A.dot(mean_param)
        if np.any(slack < 0):
            raise ValueError('point not feasible')

        value = ((np.log(slack + self.sigma) - np.log(slack)).sum() + 
                 (mean_param**2).sum() / (2 * self.sigma**2))
        grad = (-A.T.dot(1. / (slack + self.sigma) - 1. / slack) + 
                 mean_param / self.sigma**2)

        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(grad)
        elif mode == 'both':
            return self.scale(value), self.scale(grad)
        else:
            raise ValueError('mode incorrectly specified')

class gaussian_cumulant(rr.smooth_atom):

    """

    Cumulant generating function for Gaussian
    likelihood with unknown variance in the 
    regression model with design matrix $X$.

    The cumulant generating function is determined by

    .. math::

        \begin{aligned}
        e^{\Lambda(\gamma,\eta)} &= 
        (2\pi)^{-n/2} 
        \int e^{\gamma^T(X^Ty) - \frac{\eta}{2}\|y\|^2_2} \; dy \\
        &= e^{\frac{1}{2\eta}\|X\gamma\|^2_2}.
        \end{aligned}

    """

    def __init__(self, 
                 X,
                 offset=None,
                 quadratic=None,
                 initial=None):
        """

        Parameters
        ----------

        X : np.ndarray
            Design matrix.

        """
        rr.smooth_atom.__init__(self,
                                (X.shape[1] + 1,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.X = X

    def regression_parameters(self, natural_param):
        """
        From the natural parameters, form the usual
        $(\beta, \sigma^2)$ parameters.

        Parameters
        ----------

        natural_param : np.ndarray
            Natural parameters

        Returns
        -------

        The usual estimates of $(\beta, \sigma^2)$ derived
        from the natural parameters.

        """

        inv_variance = natural_param[-1] # \eta in formula above
        mean_part = natural_param[:-1]   # \gamma in formula above

        sigma_sq = 1 / inv_variance
        beta = mean_part * sigma_sq

        return beta, sigma_sq

    def smooth_objective(self, natural_param, mode='both', check_feasibility=False):
        
        natural_param = self.apply_offset(natural_param)

        X = self.X
        n = X.shape[0]

        inv_variance = natural_param[-1] # \eta in formula above
        mean_part = natural_param[:-1]   # \gamma in formula above

        pseudo_fit = X.dot(mean_part)
        value1 = ((pseudo_fit)**2).sum() / (2 * inv_variance)
        grad_var = -n / (2 * inv_variance) - value1 / inv_variance # d/d\eta
        grad_mean = X.T.dot(pseudo_fit) / inv_variance # d/d\gamma
        grad = np.hstack([grad_mean, grad_var])
        value = value1 - n / 2 * np.log(inv_variance)
        
        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(grad)
        elif mode == 'both':
            return self.scale(value), self.scale(grad)
        else:
            raise ValueError('mode incorrectly specified')

class gaussian_cumulant_conjugate(rr.smooth_atom):

    """

    Conjugate of cumulant generating function 
    for Gaussian likelihood with unknown variance 
    in the regression model with design matrix $X$.

    The cumulant generating function is determined by

    .. math::

        \begin{aligned}
        e^{\Lambda(\gamma,\eta)} &= 
        (2\pi)^{-n/2} 
        \int e^{\gamma^T(X^Ty) - \frac{\eta}{2}\|y\|^2_2} \; dy \\
        &= e^{\frac{1}{2\eta}\|X\gamma\|^2_2}.
        \end{aligned}

    The convex conjugate of this function is 

    .. math::

        \Lambda^*(\delta, s) = -\frac{n}{2}(1 - \log(n)) - \frac{n}{2} 
        \log(-2 s - \delta^T(X^TX)^{-1}\delta)

    plus the constraint $\delta \in \text{row}(X)$.

    """

    def __init__(self, 
                 X,
                 offset=None,
                 quadratic=None,
                 initial=None):
        """
        Parameters
        ----------

        X : np.ndarray
            Design matrix.

        """
        rr.smooth_atom.__init__(self,
                                (X.shape[1] + 1,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.X = X
        self._cholX = cho_factor(X.T.dot(X))

        n = X.shape[0]
        self._C = -n / 2 * (1 - np.log(n))

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
        
        mean_param = self.apply_offset(mean_param)

        X = self.X
        n = X.shape[0]

        normY_sq = -2 * mean_param[-1] # -2 s in formula above -- norm of Y^2
        mean_part = mean_param[:-1]   # \delta -- X^TY

        ### XXX We don't check this here,
        ### but mean_part should be in the row space of X!

        solve_ = cho_solve(self._cholX, mean_part)
        quadratic_part = (mean_part * solve_).sum()
        sum_sq = normY_sq - quadratic_part

        value = - n * np.log(sum_sq) / 2 + self._C

        # these parameters achieve the value of the conjugate
        # when written in variational form.
        # this means they are the gradient

        optimal_var_param = 1. / (sum_sq / n)
        optimal_mean_param = solve_ * optimal_var_param
        grad = np.hstack([optimal_mean_param, optimal_var_param])

        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(grad)
        elif mode == 'both':
            return self.scale(value), self.scale(grad)
        else:
            raise ValueError('mode incorrectly specified')

class gaussian_cumulant_known(rr.smooth_atom):

    """

    Cumulant generating function for Gaussian
    likelihood with known variance in
    the regression model with design matrix $X$.

    The cumulant generating function is

    .. math::

        \Lambda(\gamma) = \frac{\sigma^2}{2} \|X\gamma\|^2_2

    """

    def __init__(self, 
                 X,
                 sigma,
                 offset=None,
                 quadratic=None,
                 initial=None):
        """
        Parameters
        ----------

        X : np.ndarray
            Design matrix.

        sigma : float
            Known standard deviation.

        """

        rr.smooth_atom.__init__(self,
                                (X.shape[1],),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.X = X
        self.sigma = sigma

    def regression_parameters(self, natural_param):
        """
        From the natural parameters, form the usual
        $\beta$ parameters.
        """

        mean_part = natural_param   # \gamma in formula above
        beta = mean_part * self.sigma**2
        return beta

    def smooth_objective(self, natural_param, mode='both', check_feasibility=False):
        
        natural_param = self.apply_offset(natural_param)

        X = self.X
        n = X.shape[0]

        mean_part = natural_param   # \gamma in formula above

        pseudo_fit = X.dot(mean_part)
        value = self.sigma**2 * ((pseudo_fit)**2).sum() / 2.
        grad = self.sigma**2 * X.T.dot(pseudo_fit) 
        
        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(grad)
        elif mode == 'both':
            return self.scale(value), self.scale(grad)
        else:
            raise ValueError('mode incorrectly specified')

class gaussian_cumulant_conjugate_known(rr.smooth_atom):

    """

    Cumulant generating function for Gaussian
    likelihood with known variance in the regression
    model with design matrix $X$.

    The cumulant generating function is

    .. math::

        \Lambda(\gamma) = \frac{\sigma^2}{2} \|X\gamma\|^2_2

    Its conjugate is

    .. math::

        \Lambda^*(\delta) = \frac{1}{2\sigma^2} \delta^T(X^TX)^{-1}\delta

    with the constraint $\delta \in \text{row}(X)$.

    """

    def __init__(self, 
                 X,
                 sigma,
                 offset=None,
                 quadratic=None,
                 initial=None):
        """
        Parameters
        ----------

        X : np.ndarray
            Design matrix.

        sigma : float
            Known standard deviation.

        """

        rr.smooth_atom.__init__(self,
                                (X.shape[1],),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.X = X
        self.sigma = sigma
        self._cholX = cho_factor(X.T.dot(X))

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
        
        ### XXX We don't check this here,
        ### but mean_param should be in the row space of X!

        mean_param = self.apply_offset(mean_param)

        X = self.X
        n = X.shape[0]

        solve_ = cho_solve(self._cholX, mean_param)
        quadratic_part = (mean_param * solve_).sum()

        value = quadratic_part / (2 * self.sigma**2)
        grad = solve_ / self.sigma**2

        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(grad)
        elif mode == 'both':
            return self.scale(value), self.scale(grad)
        else:
            raise ValueError('mode incorrectly specified')


class optimal_tilt(rr.smooth_atom):

    """
    An objective used to find an
    approximately best tilt for a
    given affine constraint and a given
    direction of interest.

    We approximately solve the problem
    
    ..math::

        \text{min.}_{c,z:A(z + c\eta + \gamma) \leq b} \|z + c \eta\|^2_{\Sigma}

    where the objective is Mahalanobis distance
    for the constraint's covariance, $\gamma$ is
    the constraint's mean and the set
    $\{w:Aw \leq b\}$ is the affine constraint.

    """

    def __init__(self, affine_con, 
                 direction_of_interest,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                                affine_con.linear_part.shape[1] + 1,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.affine_con = affine_con
        self.direction_of_interest = eta = direction_of_interest 

        design = self.design = np.hstack([np.identity(affine_con.dim), 
                                          eta.reshape((-1,1))])

        sqrt_inv = affine_con.covariance_factors()[1]
        Si = np.dot(sqrt_inv.T, sqrt_inv)
        self.Q = np.dot(design.T, np.dot(Si, design))

        gamma = affine_con.mean

        linear_part = np.dot(affine_con.linear_part, design)
        offset = affine_con.offset - np.dot(affine_con.linear_part, 
                                            affine_con.mean)

        scaling = np.sqrt((linear_part**2).sum(1))
        linear_part /= scaling[:,None]
        offset /= scaling

        self.linear_objective = 0.

        smoothing_quadratic = rr.identity_quadratic(1.e-2, 0, 0, 0)
        self.smooth_constraint = rr.nonpositive.affine(linear_part,
                                 -offset).smoothed(
                                 smoothing_quadratic)

    def smooth_objective(self, z, mode='both', check_feasibility=False):

        Qz = np.dot(self.Q, z) 

        if mode == 'both':
            fc, gc = self.smooth_constraint.smooth_objective(z, mode='both')
            g = Qz + self.linear_objective + gc
            f = (z*Qz).sum() * 0.5 + (self.linear_objective*z).sum() + fc
            return f, g
        elif mode == 'grad':
            gc = self.smooth_constraint.smooth_objective(z, mode='grad')
            g = Qz + self.linear_objective + gc
            return g
        elif mode == 'func':
            fc = self.smooth_constraint.smooth_objective(z, mode='func')
            f = (z*Qz).sum() * 0.5 + (self.linear_objective*z).sum() + fc
            return f

    def fit(self, **regreg_args):
        soln = self.soln = self.solve(**regreg_args)
        self.z_soln = soln[:-1]
        self.c_soln = soln[-1]
        self.optimal_point = np.dot(self.design, self.soln)
        self.reweight_func = -self.affine_con.solve(self.optimal_point)
        return self.optimal_point

