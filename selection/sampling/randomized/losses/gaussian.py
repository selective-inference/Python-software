import numpy as np
from base import selective_loss

class gaussian_Xfixed(selective_loss):

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
        self._restricted_grad_beta = np.zeros(self.shape)

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
        old_data, self.y = self.y, data
        g = self.smooth_objective(beta, 'grad')
        self.y = old_data
        return g

    def hessian(self, data, beta):
        if not hasattr(self, "_XTX"):
            self._XTX = np.dot(self.X.T, self.X)
        return self._XTX

    def setup_sampling(self, y, mean, sigma, linear_part, value):

        ### JT: if sigma is known the variance should be adjusted
        ### if it is unknown then the pdf below should be uniform
        ### supported on sphere of some radius

        ### This can be implemented as part of 
        ### a subclass

        self.accept_data = 0
        self.total_data = 0


        self.sigma = sigma

        P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
        I = np.identity(linear_part.shape[1])

        self.data = y
        self.mean = mean

        self.R = I - P


        self.P = P
        self.linear_part = linear_part

    def proposal(self, data):
        n, p = self.X.shape
        stepsize = 2. / np.sqrt(n)
        new = data + stepsize * np.dot(self.R, 
                                       self.sigma * np.random.standard_normal(n))
        log_transition_p = self.logpdf(new) - self.logpdf(data)
        return new, log_transition_p

    def logpdf(self, y):
        ### Up to some constant...
        return -((y - self.mean)**2).sum() / (2 * self.sigma**2)

    def update_proposal(self, state, proposal, logpdf):
        pass


class sqrt_Lasso_Xfixed(gaussian_Xfixed):
    
    ### linear part is X_{E\j}^T
    def proposal(self, y):
        P, R = self.R, self.P
        residual = np.dot(R, y)

        eta = np.dot(R, np.random.standard_normal(y.shape))
        eta -= np.dot(residual, eta) * residual / (np.linalg.norm(residual)**2)
        eta /= np.linalg.norm(eta)

        # we should try adaptive MCMC on this to get a proper scale

        n = self.R.shape[0]
        theta = np.random.beta(1, n/4)
        new_sample = np.cos(theta) * residual + np.sin(theta) * np.linalg.norm(residual) * eta + np.dot(P, y)

        log_transition_p = 0
        return new_sample, log_transition_p

    def gradient(self, data, beta):
        old_data, self.y = self.y, data
        residual = self.y - np.dot(self.X, beta)
        g = - np.dot(self.X.T, residual) / np.linalg.norm(residual)
        self.y = old_data
        return g

    ### calculate the det part in sqrt lasso
    ### selected_part is X_E^T

    def hessian(self, data, beta):

        selected_part = self.X.T
        n = selected_part.shape[1]
        residual = data - np.dot(self.X, beta)
        R = np.identity(n) - np.outer(residual, residual) / (np.linalg.norm(residual)**2)
        temp = np.dot(selected_part, R)
        result = np.dot(temp, selected_part.T) / (np.linalg.norm(residual)**2) 
        return result
