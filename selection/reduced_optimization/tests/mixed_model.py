import numpy as np

class instance_mixed(object):

    def __init__(self, n, p, s, sigma=1., rho=0, random_signs=False, scale =True, center=True):
         (self.n, self.p, self.s,
         self.sigma,
         self.rho) = (n, p, s,
                     sigma,
                     rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
              np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         self.beta = np.zeros(p)
         self.beta[:self.s] = np.linspace(0.5, 5.0, num=s)
         if random_signs:
             self.beta[:self.s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
         self.active = np.zeros(p, np.bool)
         self.active[:self.s] = True

    def _noise(self):
        return np.random.standard_normal(self.n)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + self._noise()) * self.sigma
        return self.X, Y, self.beta * self.sigma, np.nonzero(self.active)[0], self.sigma
