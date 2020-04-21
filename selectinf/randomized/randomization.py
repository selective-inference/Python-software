"""
Different randomization options for selective sampler.
Main method used in selective sampler is the gradient method which
should be a gradient of the negative of the log-density. For a
Gaussian density, this will be a convex function, not a concave function.
"""
from __future__ import division, print_function
import numpy as np
import regreg.api as rr
from scipy.stats import laplace, logistic, norm as ndist

class randomization(rr.smooth_atom):

    def __init__(self,
                 shape,
                 density,
                 cdf,
                 pdf,
                 derivative_log_density,
                 grad_negative_log_density,
                 sampler,
                 lipschitz=1,
                 log_density=None,
                 CGF=None,  # cumulant generating function and gradient
                 CGF_conjugate=None,  # convex conjugate of CGF and gradient
                 cov_prec=None      # will have a covariance matrix if Gaussian
                 ):

        rr.smooth_atom.__init__(self,
                                shape)
        self._density = density
        self._cdf = cdf
        self._pdf = pdf
        self._derivative_log_density = derivative_log_density
        self._grad_negative_log_density = grad_negative_log_density
        self._sampler = sampler
        self.lipschitz = lipschitz

        if log_density is None:
            log_density = lambda x: np.log(density(x))

        self._log_density = log_density
        self.CGF = CGF
        self.CGF_conjugate = CGF_conjugate
        if cov_prec is not None:
            self.cov_prec = cov_prec

    def smooth_objective(self, perturbation, mode='both', check_feasibility=False):
        """
        Compute the negative log-density and its gradient.
        """
        if mode == 'func':
            return self.scale(-np.log(self._density(perturbation)))
        elif mode == 'grad':
            return self.scale(self._grad_negative_log_density(perturbation))
        elif mode == 'both':
            return self.scale(-np.log(self._density(perturbation))), self.scale(self._grad_negative_log_density(perturbation))
        else:
            raise ValueError("mode incorrectly specified")

    def sample(self, size=()):
        return self._sampler(size=size)

    def gradient(self, perturbation):
        """
        Evaluate the gradient of the log-density.
        Parameters
        ----------
        perturbation : np.float
        Returns
        -------
        gradient : np.float
        """
        return self.smooth_objective(perturbation, mode='grad')

    def log_density(self, perturbation):
        """
        Evaluate the log-density.
        Parameters
        ----------
        perturbation : np.float
        Returns
        -------
        value : float
        """
        return np.squeeze(self._log_density(perturbation))

    def randomize(self, loss, epsilon=0, perturb=None):
        """
        Randomize the loss.
        """
        randomized_loss = rr.smooth_sum([loss])
        if perturb is None:
            perturb = self.sample()
        randomized_loss.quadratic = rr.identity_quadratic(epsilon, 0, -perturb, 0)
        return randomized_loss, perturb

    @staticmethod
    def isotropic_gaussian(shape, scale):
        """
        Isotropic Gaussian with SD `scale`.
        Parameters
        ----------
        shape : tuple
            Shape of noise.
        scale : float
            SD of noise.
        """
        if type(shape) == type(1):
            shape = (shape,)
        rv = ndist(scale=scale, loc=0.)
        p = np.product(shape)
        density = lambda x: np.product(rv.pdf(x))
        cdf = lambda x: rv.cdf(x)
        pdf = lambda x: rv.pdf(x)
        derivative_log_density = lambda x: -x/(scale**2)
        grad_negative_log_density = lambda x: x / scale**2
        sampler = lambda size: rv.rvs(size=shape + size)
        CGF = isotropic_gaussian_CGF(shape, scale)
        CGF_conjugate = isotropic_gaussian_CGF_conjugate(shape, scale)

        p = np.product(shape)
        constant = 0 # -0.5 * p * np.log(2 * np.pi * scale**2)
        return randomization(shape,
                             density,
                             cdf,
                             pdf,
                             derivative_log_density,
                             grad_negative_log_density,
                             sampler,
                             lipschitz=1./scale**2,
                             log_density = lambda x: -0.5 * (np.atleast_2d(x)**2).sum(1) / scale**2 + constant,
                             CGF=CGF,
                             CGF_conjugate=CGF_conjugate,
                             cov_prec=(scale**2, 1. / scale**2)
                             )

    @staticmethod
    def gaussian(covariance):
        """
        Gaussian noise with a given covariance.
        Parameters
        ----------
        covariance : np.float((*,*))
            Positive definite covariance matrix. Non-negative definite
            will raise an error.
        """
        precision = np.linalg.inv(covariance)
        sqrt_precision = np.linalg.cholesky(precision)
        _det = np.linalg.det(covariance)
        p = covariance.shape[0]
        _const = 1. # np.sqrt((2*np.pi)**p * _det)
        density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        cdf = lambda x: None
        pdf = lambda x: None
        derivative_log_density = lambda x: None
        grad_negative_log_density = lambda x: precision.dot(x)
        sampler = lambda size: covariance.dot(sqrt_precision.dot(np.random.standard_normal((p,) + size)))

        return randomization((p,),
                             density,
                             cdf,
                             pdf,
                             derivative_log_density,
                             grad_negative_log_density,
                             sampler,
                             lipschitz=np.linalg.svd(precision)[1].max(),
                             log_density = lambda x: -np.sum(sqrt_precision.dot(np.atleast_2d(x).T)**2, 0) * 0.5 - np.log(_const),
                             cov_prec=(covariance, precision))

    @staticmethod
    def degenerate_gaussian(covariance, tol=1.e-6):
        """
        Gaussian noise with a given covariance.
        Parameters
        ----------
        covariance : np.float((*,*))
            Positive definite covariance matrix. Non-negative definite
            will raise an error.
        """
        p = covariance.shape[0]
        U, D, _ = np.linalg.svd(covariance)
        keep = D > D.max() * tol
        rank = keep.sum()
        sqrt_cov = U[:,keep].dot(np.diag(np.sqrt(D[keep])))
        sqrt_precision = U[:,keep].dot(np.diag(1./np.sqrt(D[keep])))
        precision = sqrt_precision.dot(sqrt_precision.T)
        _const = 1. 
        density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        cdf = lambda x: None
        pdf = lambda x: None
        derivative_log_density = lambda x: None
        grad_negative_log_density = lambda x: precision.dot(x)
        sampler = lambda size: covariance.dot(sqrt_precision.dot(np.random.standard_normal((rank,) + size)))

        return randomization((p,),
                             density,
                             cdf,
                             pdf,
                             derivative_log_density,
                             grad_negative_log_density,
                             sampler,
                             lipschitz=(1/D[keep]).max(),
                             log_density = lambda x: -np.sum(sqrt_precision.T.dot(np.atleast_2d(x).T)**2, 0) * 0.5 - np.log(_const),
                             cov_prec=(covariance, precision))



    @staticmethod
    def laplace(shape, scale):
        """
        Standard Laplace noise multiplied by `scale`
        Parameters
        ----------
        shape : tuple
            Shape of noise.
        scale : float
            Scale of noise.
        """
        rv = laplace(scale=scale, loc=0.)
        density = lambda x: np.product(rv.pdf(x))

        grad_negative_log_density = lambda x: np.sign(x) / scale
        sampler = lambda size: rv.rvs(size=shape + size)
        cdf = lambda x: laplace.cdf(x, loc=0., scale = scale)
        pdf = lambda x: laplace.pdf(x, loc=0., scale = scale)
        derivative_log_density = lambda x: -np.sign(x)/scale
        grad_negative_log_density = lambda x: np.sign(x) / scale
        sampler = lambda size: rv.rvs(size=shape + size)
        CGF = laplace_CGF(shape, scale)
        CGF_conjugate = laplace_CGF_conjugate(shape, scale)
        constant = -np.product(shape) * np.log(2 * scale)

        return randomization(shape,
                             density,
                             cdf,
                             pdf,
                             derivative_log_density,
                             grad_negative_log_density,
                             sampler,
                             lipschitz=1./scale**2,
                             log_density = lambda x: -np.fabs(np.atleast_2d(x)).sum(1) / scale - np.log(scale) + constant,
                             CGF=CGF,
                             CGF_conjugate=CGF_conjugate,)

    @staticmethod
    def logistic(shape, scale):
        """
        Standard logistic noise multiplied by `scale`
        Parameters
        ----------
        shape : tuple
            Shape of noise.
        scale : float
            Scale of noise.
        """
        # from http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.logistic.html
        density = lambda x: (np.product(np.exp(-x / scale) /
                                        (1 + np.exp(-x / scale))**2)
                             / scale**(np.product(x.shape)))
        cdf = lambda x: logistic.cdf(x, loc=0., scale = scale)
        pdf = lambda x: logistic.pdf(x, loc=0., scale = scale)
        derivative_log_density = lambda x: (np.exp(-x/scale)-1)/(scale*np.exp(-x/scale)+1)
        # negative log density is (with \mu=0)
        # x/s + log(s) + 2 \log (1 + e(-x/s))
        grad_negative_log_density = lambda x: (1 - np.exp(-x / scale)) / ((1 + np.exp(-x / scale)) * scale)
        sampler = lambda size: np.random.logistic(loc=0, scale=scale, size=shape + size)

        constant = - np.product(shape) * np.log(scale)
        return randomization(shape,
                             density,
                             cdf,
                             pdf,
                             derivative_log_density,
                             grad_negative_log_density,
                             sampler,
                             lipschitz=.25/scale**2,
                             log_density = lambda x: -np.atleast_2d(x).sum(1) / scale - 2 * np.log(1 + np.exp(-np.atleast_2d(x) / scale)).sum(1) + constant)

class split(randomization):

    def __init__(self, shape, subsample_size, total_size):

        self.subsample_size = subsample_size
        self.total_size = total_size

        rr.smooth_atom.__init__(self,
                                shape)

    def get_covariance(self):
        if hasattr(self, "_covariance"):
            return self._covariance

    def set_covariance(self, covariance):
        """
        Once covariance has been set, then
        the usual API of randomization will work.
        """
        self._covariance = covariance
        precision = np.linalg.inv(covariance)
        self._cov_prec = (covariance, precision)
        sqrt_precision = np.linalg.cholesky(precision).T
        _det = np.linalg.det(covariance)
        p = covariance.shape[0]
        _const = 1. # np.sqrt((2*np.pi)**p * _det)
        self._density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        self._grad_negative_log_density = lambda x: precision.dot(x)
        self._sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
        self.lipschitz = np.linalg.svd(precision)[1].max()
        def _log_density(x):
            return -np.sum(sqrt_precision.dot(np.atleast_2d(x).T)**2, 0) * 0.5 - np.log(_const)
        self._log_density = _log_density

    covariance = property(get_covariance, set_covariance)

    @property
    def cov_prec(self):
        if hasattr(self, "_covariance"):
            return self._cov_prec

    def smooth_objective(self, perturbation, mode='both', check_feasibility=False):
        if not hasattr(self, "_covariance"):
            raise ValueError('first set the covariance')
        return randomization.smooth_objective(self, perturbation, mode=mode, check_feasibility=check_feasibility)

    def sample(self, size=()):
        if not hasattr(self, "_covariance"):
            raise ValueError('first set the covariance')
        return randomization.sample(self, size=size)

    def gradient(self, perturbation):
        if not hasattr(self, "_covariance"):
            raise ValueError('first set the covariance')
        return randomization.gradient(self, perturbation)

    def randomize(self, loss, epsilon):
        """
        Parameters
        ----------
        loss : rr.glm
            A glm loss with a `subsample` method.
        epsilon : float
            Coefficient in front of quadratic term
        Returns
        -------
        Subsampled loss multiplied by `n / m` where
        m is the subsample size out of a total
        sample size of n.
        The quadratic term is not multiplied by `n / m`
        """
        n, m = self.total_size, self.subsample_size
        inv_frac = n / m
        quadratic = rr.identity_quadratic(epsilon, 0, 0, 0)
        m, n = self.subsample_size, self.total_size # shorthand
        idx = np.zeros(n, np.bool)
        idx[:m] = 1
        np.random.shuffle(idx)

        randomized_loss = loss.subsample(idx)
        randomized_loss.coef *= inv_frac

        randomized_loss.quadratic = quadratic

        return randomized_loss, None

# Conjugate generating function for Gaussian

def isotropic_gaussian_CGF(shape, scale): # scale = SD
    return cumulant(shape,
                    lambda x: (x**2).sum() * scale**2 / 2.,
                    lambda x: scale**2 * x)

def isotropic_gaussian_CGF_conjugate(shape, scale):  # scale = SD
    return cumulant_conjugate(shape,
                              lambda x: (x**2).sum() / (2 * scale**2),
                              lambda x: x / scale**2)

# Conjugate generating function for Laplace

def _standard_laplace_CGF_conjugate(u):
    """
    sup_z uz + log(1 - z**2)
    """
    _zeros = (u == 0)
    root = (-1 + np.sqrt(1 + u**2)) / (u + _zeros)
    value = (root * u + np.log(1 - root**2)).sum()
    return value

def _standard_laplace_CGF_conjugate_grad(u):
    """
    sup_z uz + log(1 - z**2)
    """
    _zeros = (u == 0)
    root = (-1 + np.sqrt(1 + u**2)) / (u + _zeros)
    return root

BIG = 10**10
def laplace_CGF(shape, scale):
    return cumulant(shape,
                    lambda x: -np.log(1 - (scale * x)**2).sum() + BIG * (np.abs(x) > 1),
                    lambda x: 2 * x * scale**2 / (1 - (scale * x)**2))

def laplace_CGF_conjugate(shape, scale):
    return cumulant_conjugate(shape,
                              lambda x: _standard_laplace_CGF_conjugate(x / scale),
                              lambda x: _standard_laplace_CGF_conjugate_grad(x / scale) / scale)

class from_grad_func(rr.smooth_atom):

    """
    take a (func, grad) pair and make a smooth_objective
    """


    def __init__(self,
                 shape,
                 func,
                 grad,
                 coef=1.,
                 offset=None,
                 initial=None,
                 quadratic=None):

        rr.smooth_atom.__init__(self,
                                shape,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self._func, self._grad = (func, grad)

    def smooth_objective(self, param, mode='both', check_feasibility=False):
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        param = self.apply_offset(param)

        if mode == 'func':
            return self.scale(self._func(param))
        elif mode == 'grad':
            return self.scale(self._grad(param))
        elif mode == 'both':
            return self.scale(self._func(param)), self.scale(self._grad(param))
        else:
            raise ValueError("mode incorrectly specified")


class cumulant(from_grad_func):
    """
    Class for CGF.
    """
    pass

class cumulant_conjugate(from_grad_func):
    """
    Class for conjugate of a CGF.
    """
    pass

