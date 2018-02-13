import regreg.api as rr
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm
from selection.randomized.api import randomization as rm
from scipy.stats import norm, laplace, gaussian_kde, f, chi2

from scipy.optimize import bisect
import MH as mh

# this file uses basics in MH.py to 
# use tsls statistic as the sampling variable
# note this is different from using tsls estimator
# compute the selective CIs using importance sampling
# simulate the coverage of CIs
# refer to code in selectice package: selection.randomized.query.py -> optimization_intervals


# tsls consistent estimator of beta^*
def beta_tsls(Z, D, Y, E):
    n, p = Z.shape
    P_Z = Z.dot(np.linalg.pinv(Z))
    P_ZE = Z[:, E].dot(np.linalg.pinv(Z[:, E]))
    P_diff = P_Z - P_ZE
    beta_estim = (D.T.dot(P_diff).dot(Y)) / (D.T.dot(P_diff).dot(D))
    return beta_estim


# assume when init, beta_reference is beta_tsls
class tilted_CI(object):

	def __init__(self):
		pass

	def tilted_confidence_interval(self, Z, D, Y, lagrange, randomizer, epsilon,
                                   E, s_E, beta, alpha_E, u_nE, randomization_sampling,
                                   b0_leftStart, b0_rightStart,
                                   true_model=True, Sigma = None, alpha = 0.05,
								   ndraw=10000, burnin=5000, data_stepsize=6., coefs_stepsize=0.001, 
								   samples=None):
		if true_model is True:
			Sigmab0 = Sigma
		else:
			raise ValueError('need to implement for unknown covariance!')

		self.beta_reference = beta_tsls(Z, D, Y, E)

		sampler = mh.MH_sampler(Z, D, Y, lagrange, randomizer, epsilon, E, s_E, beta, alpha_E, u_nE, 
								self.beta_reference, Sigmab0, randomization_sampling, data_stepsize=data_stepsize, coefs_stepsize=coefs_stepsize)

		if samples is None:
			samples = sampler.sample(ndraw, burnin, True)

		#print 'samples,', samples.shape

		self._logden = self.density_factor(sampler, samples, self.beta_reference)
		#print 'logden,', self._logden

		grid_min, grid_max = b0_leftStart - self.beta_reference, b0_rightStart - self.beta_reference
		#print 'grid, ', grid_min, grid_max

		def _rootU(gamma):
			return self.pivot(sampler, samples, self.beta_reference + gamma) - (alpha)/2.
		def _rootL(gamma):
			return self.pivot(sampler, samples, self.beta_reference + gamma) - (2.-alpha)/2.

		upper = bisect(_rootU, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))
		lower = bisect(_rootL, grid_min, grid_max, xtol=1.e-5*(grid_max - grid_min))

		return self.beta_reference + lower, self.beta_reference + upper, upper - lower


	# b is the current b0 value that we want to estimate the samples at
	# always return the pivot value for onesided 'less'
	def pivot(self, sampler, samples, b):
		#sample_stat = samples[:, sampler.data_slice] + (b - self.beta_reference)
		sample_stat = samples[:, sampler.data_slice]
		# compute T_obs at b
		observed_stat = sampler.K32 - sampler.K33 * b
		weights = self._weights(sampler, samples, b)

		pivot = np.mean((sample_stat <= observed_stat) * weights) / np.mean(weights)
		#print 'pivot,', pivot
 
		return pivot

	# the density factor that matters when computing the importance ratio
	# in here it is g(b) part
	def density_factor(self, sampler, samples, b):
		#_logratio = None
		#T_samples = np.squeeze(samples[:, sampler.state_Tindex])
		#_logratio = 0.5*((T_samples - self.beta_reference)**2-(T_samples - b)**2)
		#result = sampler.randomizer.log_density(sampler.tilted_reconstruction_map(samples, b))
		logden = []
		for state in samples:
			logden.append(sampler.randomizer.log_density(sampler.tilted_reconstruction_map(state, b)))
		return np.array(logden)


	# private methods
	# hard code for now
	def _weights(self, sampler, samples, b):
		_lognum = self.density_factor(sampler, samples, b)
		_logratio = _lognum - self._logden
		#_logratio = self.density_factor(sampler, samples, b)
		_logratio -= _logratio.max()

		return np.exp(_logratio)


