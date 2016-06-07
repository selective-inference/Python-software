import numpy as np
from base import selective_loss

class gaussian_Xfixed_high_dim(selective_loss):

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
        #self._restricted_grad_beta = np.zeros(self.shape)

    def smooth_objective(self, beta, mode='both',
                         check_feasibility=False):


        resid = self.y - np.dot(self.X, beta)

        if mode == 'both':
            f = self.scale((resid**2).sum()) / 2.
            g = self.scale(-np.dot(self.X.T, resid))
            return f, g
        elif mode == 'func':
            f = self.scale(np.linalg.norm(resid)**2) / 2.
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
        #old_data, self.y = self.y, data
        #g = self.smooth_objective(beta, 'grad')
        #self.y = old_data
        X = self.X
        hessian = np.dot(X.T,X)
        return -(data+np.dot(hessian, beta))

    def hessian(self): #, data, beta):
        if not hasattr(self, "_XTX"):
            self._XTX = np.dot(self.X.T, self.X)
        return self._XTX

    def setup_sampling(self, data):

        ### JT: if sigma is known the variance should be adjusted
        ### if it is unknown then the pdf below should be uniform
        ### supported on sphere of some radius

        ### This can be implemented as part of
        ### a subclass

        self.accept_data = 0
        self.total_data = 0


        #self.sigma = sigma


        self.data = data
        #self.mean = mean


    def proposal(self, data):
        n, p = self.X.shape
        stepsize = 4. / np.sqrt(n)  # originally 2. / np.sqrt(n)

        new = data + stepsize * np.dot(self.R,
                                       self.sigma * np.random.standard_normal(n))


        #stepsize = 5./n
        #sign_vector =  np.sign(val)

        #grad_log_pi = -(data + np.dot(self.X,sign_vector))

        #grad_log_pi = 0

        #new = data + np.dot(self.R,
        #                    (stepsize*grad_log_pi) + (np.sqrt(2*stepsize)*np.random.standard_normal(data.shape[0])))

        log_transition_p = self.logpdf(new) - self.logpdf(data)

        return new, log_transition_p


    def logpdf(self, y):
        ### Up to some constant...
        return -((y - self.mean)**2).sum() / (2 * self.sigma**2)

    def update_proposal(self, state, proposal, logpdf):
        pass

