import numpy as np
from base import selective_loss
from scipy.sparse.csgraph import connected_components

class neighbourhood_selection(selective_loss):

    def __init__(self, X,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None):
        p = X.shape[1]
        selective_loss.__init__(self, p**2 - p,
                                coef=coef,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)
        """
        X.shape = (n,p)
        """

        self.X = X.copy()
        self.off_diagonal = ~np.identity(p, dtype=bool)

    def smooth_objective(self, beta, mode='both',
                         check_feasibility=False):
        """
        beta.shape = (p^2 - p,) 
        """
        resid = self.X - np.dot(self.X, self.reshape(beta))
        
        if mode == 'both':
            f = self.scale((resid**2).sum()) / 2.
            g = self.scale(-np.dot(self.X.T, resid)[self.off_diagonal])
            return f, g
        elif mode == 'func':
            f = self.scale((resid**2).sum()) / 2.
            return f
        elif mode == 'grad':
            g = self.scale(-np.dot(self.X.T, resid)[self.off_diagonal])
            return g
        else:
            raise ValueError("mode incorrectly specified")

    # this is something that regreg does not know about, i.e.
    # what is data and what is not...

    def reshape(self, beta):
        p = self.X.shape[1]
        B = np.zeros((p, p))
        B[self.off_diagonal] = beta
        return B 

    def gradient(self, data, beta):
        """
        Gradient of smooth part restricted to active set
        """

        old_data, self.X = self.X, data
        g = self.smooth_objective(beta, 'grad')
        self.X = old_data
        return g

    def hessian(self, data, beta):
        return np.identity(self.shape[0]) 

    def setup_sampling(self, X, active, quadratic_coef):

        self.accept_data = 0
        self.total_data = 0

        self.quadratic_coef = quadratic_coef
        self.data = X 

        self.edges = self.reshape(active).astype(bool)
        self.graph = np.logical_or(self.edges.T, self.edges)

        if not hasattr(self, "ncomponents") or not hasattr(self, "labels"):
            self.ncomponents, self.labels = \
                    connected_components(self.graph, directed=False)

    def proposal(self, data):
        """
        update one column of X at a time
        """
        # pick one random column
        n, p = data.shape
        idx = np.random.choice(range(p)) 
        keep = (self.labels == self.labels[idx]) 
        keep[idx] = False

        # compute the projection matrix
        if keep.any():
            L = data[:, keep] 
            P = np.dot(L, np.linalg.pinv(L))
        else:
            P = np.zeros((n, n))
        R = np.identity(n) - P

        # compute the proposal 
        residual = np.dot(R, data[:, idx])
        eta = np.dot(R, np.random.standard_normal(n))
        eta -= np.dot(residual, eta) * residual / (np.linalg.norm(residual)**2)
        eta /= np.linalg.norm(eta)
        theta = np.pi * np.random.beta(1, 1)
        new_col = np.cos(theta) * residual + np.sin(theta) * np.linalg.norm(residual) \
                * eta + np.dot(P, data[:, idx])
        new = data.copy()
        new[:, idx] = new_col
        J_new = np.dot(new.T, new) + self.quadratic_coef * np.identity(p)
        J = np.dot(data.T, data) + self.quadratic_coef * np.identity(p)

        # update the determinant after changing one column
        log_transition_p = 0
        nbs = np.where(self.edges[:, idx])[0]
        for i in nbs:
            active_i = self.edges[:, i] 
            if active_i[idx]:
                log_transition_p += self.logdet_ratio(J_new, J, active_i)

        return new, log_transition_p

    def logpdf(self, y):
        pass

    def update_proposal(self, state, proposal, logpdf):
        pass


    def logdet_ratio(self, new, old, active):
        new_E = new[active, :][:, active]
        old_E = old[active, :][:, active] 
        return np.linalg.slogdet(new_E)[1] - np.linalg.slogdet(old_E)[1]

