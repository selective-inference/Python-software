import numpy as np
from scipy.stats import norm as ndist

# randomization mechanism

class normal_sampler(object):

    """
    Our basic model for noise, and input to 
    selection algorithms. This represents
    Gaussian data with a center, e.g. X.T.dot(y)
    in linear regression and a covariance Sigma.

    This object emits noisy versions of `center` as

    center + scale * N(0, Sigma)

    """
    def __init__(self, center, covariance):
        '''
        Parameters
        ----------

        center : np.float(p)
            Center of Gaussian noise source.

        covariance : np.float((p, p))
            Covariance of noise added (up to scale factor).

        '''
        (self.center,
         self.covariance) = (np.asarray(center),
                             np.asarray(covariance))
        self.shape = self.center.shape

    def __call__(self, size=None, scale=1.):

        '''

        Parameters
        ----------

        size : tuple or int
            How many copies to draw

        scale : float
            Scale (in data units) applied to unitless noise before adding.

        Returns
        -------

        noisy_sample : np.float

        Generate noisy version of the center. With scale==0.,
        return the full center.

        TODO: for some calculations, a log of each call would be helpful
        for constructing UMVU, say.

        '''

        if not hasattr(self, 'cholT'):
            self.cholT = np.linalg.cholesky(self.covariance).T

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        if self.shape == ():
            _shape = (1,)
        else:
            _shape = self.shape
        return scale * np.squeeze(np.random.standard_normal(size + _shape).dot(self.cholT)) + self.center

class split_sampler(object):

    """
    Data splitting noise source.
    This is approximately
    Gaussian with center np.sum(sample_stat, 0)
    and noise suitably scaled, depending
    on splitting fraction.

    This object emits noisy versions of `center` as

    center + scale * N(0, Sigma)

    """

    def __init__(self, sample_stat, covariance): # covariance of sum of rows
        '''
        Parameters
        ----------

        sample_stat : np.float((n, p))
             Data matrix. In linear regression this is X * y[:, None]

        covariance : np.float((p, p))
             Covariance of np.sum(sample_stat, 0). Could be computed
             e.g. by bootstrap or parametric method given a design X.
        '''
        self.sample_stat = np.asarray(sample_stat)
        self.nsample = self.sample_stat.shape[0]
        self.center = np.sum(self.sample_stat, 0)
        self.covariance = covariance
        self.shape = self.center.shape

    def __call__(self, size=None, scale=0.5):

        '''
        Parameters
        ----------

        size : tuple or int
            How many copies to draw

        scale : float
            Scale (in data units) applied to unitless noise before adding.

        Returns
        -------

        noisy_sample : np.float

        Generate noisy version of the center. With scale==0.,
        return the full center.

        The equivalent data splitting fraction is 1 / (scale**2 + 1).
        Argument is kept as `scale` instead of `frac` so that the general
        learning algorithm can replace this `splitter_source` with a corresponding
        `normal_source`.

        TODO: for some calculations, a log of each call would be helpful
        for constructing UMVU, say.
        '''

        # (1 - frac) / frac = scale**2

        frac = 1 / (scale**2 + 1)

        if type(size) == type(1):
            size = (size,)
        size = size or (1,)
        if self.shape == ():
            _shape = (1,)
        else:
            _shape = self.shape

        final_sample = []
        idx = np.arange(self.nsample)
        for _ in range(np.product(size)):
            sample_ = self.sample_stat[np.random.choice(idx, int(frac * self.nsample), replace=False)]
            final_sample.append(np.sum(sample_, 0) / frac) # rescale to the scale of a sum of nsample rows
        val = np.squeeze(np.array(final_sample).reshape(size + _shape))
        return val

