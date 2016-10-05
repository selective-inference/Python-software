"""
Different randomization options for selective sampler.

Main method used in selective sampler is the gradient method which
should be a gradient of the negative of the log-density. For a 
Gaussian density, this will be a convex function, not a concave function.
"""

import numpy as np
import regreg.api as rr
from scipy.stats import laplace, norm as ndist

class randomization(rr.smooth_atom):

    def __init__(self, shape, density, grad_negative_log_density, sampler, lipschitz=1):
        rr.smooth_atom.__init__(self,
                                shape)
        self._density = density
        self._grad_negative_log_density = grad_negative_log_density
        self._sampler = sampler
        self.lipschitz = lipschitz

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

    @staticmethod
    def isotropic_gaussian(shape, scale):
        rv = ndist(scale=scale, loc=0.)
        density = lambda x: rv.pdf(x)
        grad_negative_log_density = lambda x: x / scale**2
        sampler = lambda size: rv.rvs(size=shape + size)
        return randomization(shape, density, grad_negative_log_density, sampler, lipschitz=1./scale**2)

    @staticmethod
    def gaussian(covariance):
        precision = np.linalg.inv(covariance)
        sqrt_precision = np.linalg.cholesky(precision)
        _det = np.linalg.det(covariance)
        p = covariance.shape[0]
        _const = np.sqrt((2*np.pi)**p * _det)
        density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        grad_negative_log_density = lambda x: precision.dot(x)
        sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
        return randomization((p,), density, grad_negative_log_density, sampler, lipschitz=np.linalg.svd(precision)[1].max())

    @staticmethod
    def laplace(shape, scale):
        rv = laplace(scale=scale, loc=0.)
        density = lambda x: rv.pdf(x)
        grad_negative_log_density = lambda x: np.sign(x) / scale
        sampler = lambda size: rv.rvs(size=shape + size)
        return randomization(shape, density, grad_negative_log_density, sampler, lipschitz=1./scale**2)

    @staticmethod
    def logistic(shape, scale):
        # from http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.logistic.html
        density = lambda x: (np.exp(-x / scale) / (1 + np.exp(-x / scale))**2) / scale
        # negative log density is (with \mu=0)
        # x/s + log(s) + 2 \log (1 + e(-x/s))
        grad_negative_log_density = lambda x: (1 - np.exp(-x / scale)) / ((1 + np.exp(-x / scale)) * scale)
        sampler = lambda size: np.random.logistic(loc=0, scale=scale, size=shape + size)
        return randomization(shape, density, grad_negative_log_density, sampler, lipschitz=.25/scale**2)


class split(randomization):

    def __init__(self,X,Y,m):

        n, p = X.shape

        def _boot_covariance(indices):
            X_star = X[indices]
            Y_star = Y[indices]
            subsample = np.random.choice(n, size=(m,), replace=True)
            result = np.dot(X_star[subsample].T, Y_star[subsample]) - ((m/float(n))*np.dot(X_star.T, Y_star))
            return result

        def _nonparametric_covariance_estimate(nboot=10000):
            results = []
            for i in range(nboot):
                indices = np.random.choice(n, size=(n,), replace=True)
                results.append(_boot_covariance(indices))

            mean_results = np.zeros(p)
            for i in range(nboot):
                mean_results = np.add(mean_results, results[i])

            mean_results /= nboot

            covariance = np.zeros((p,p))
            for i in range(nboot):
                covariance = np.add(covariance, np.outer(results[i]-mean_results, results[i]-mean_results))

            return covariance/float(nboot)

        self.covariance = _nonparametric_covariance_estimate()
        print np.diag(self.covariance)
        #covariance_inv = np.linalg.inv(self.covariance)

        #from scipy.stats import multivariate_normal

        #density = lambda x: multivariate_normal.pdf(x, mean=np.zeros(p), cov=self.covariance)
        #grad_negative_log_density = lambda x: covariance_inv.dot(x)
        #sampler = lambda size: np.random.multivariate_normal(mean=np.zeros(p), cov=self.covariance, size=size)

        def gaussian(covariance):
            precision = np.linalg.inv(covariance)
            sqrt_precision = np.linalg.cholesky(precision)
            _det = np.linalg.det(covariance)
            p = covariance.shape[0]
            _const = np.sqrt((2 * np.pi) ** p * _det)
            density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
            grad_negative_log_density = lambda x: precision.dot(x)
            sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
            return randomization.__init__(self,(p,), density, grad_negative_log_density, sampler,
                                 lipschitz=np.linalg.svd(precision)[1].max())

        gaussian(self.covariance*5)
        #randomization.__init__(self, 1, density, grad_negative_log_density, sampler)




