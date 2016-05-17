import numpy as np
from base import selective_loss
#from regreg.smooth.glm import logistic_loss

class lasso_randomX(selective_loss):

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

        self.X = X.copy()
        self._XTX = np.dot(self.X.T,self.X).copy()

        self.y = y.copy()


    def smooth_objective(self, beta, mode='both',
                         check_feasibility=False):
        """
        smooth_objective used to initialize gradient in sampler.py
        only uses 'grad' for the mode
        """

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


        # _loss = logistic_loss(self.X, self.y, coef=self.X.shape[0]/2.)
        #
        # return _loss.smooth_objective(beta, mode=mode, check_feasibility=check_feasibility)

    # this is something that regreg does not know about, i.e.
    # what is data and what is not...

    def fit_E(self, active, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Computes the OLS estimator \bar{\beta}_E (y~X_E) after seeing the active set (E).
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.

        Parameters:
        ----------
        active: the active set from fitting the randomized lasso for the first time

        solve_args: passed to regreg.simple_problem.solve # not used here

        """

        self.active = active

        if self.active.any():
            self.inactive = ~active
            X_E = self.X[:, self.active]
            self.size_active = X_E.shape[1] # |E|

            self._XTX = np.dot(self.X.T,self.X)
            self._XETXE = np.dot(X_E.T,X_E)

            self._XTXE = np.dot(self.X.T, X_E)

            #loss_E = logistic_loss(X_E, self.y)
            #self._beta_unpenalized = loss_E.solve(**solve_args)
            self._beta_unpenalized = np.linalg.lstsq(X_E, self.y)[0]  # \bar{\beta}_E
            self.bootstrap_covariance()
        else:
            raise ValueError("Empty active set.")


    def bootstrap_covariance(self):
        """
        Computes the following covariances:

        self._cov_XTepsilon : bootstrapped covariance matrix of X^T\epsilon,
                              where \epsilon = y-X_E\beta_E (\beta_E true underlying parameter)

        self._cov_XETepsilon : bootstrapped covariance matrix of X_{E}^T\epsilon

        self._cov_N : bootstrapped covariance matrix of the null statistic
                        X_{-E}^T(y-X_E\bar{\beta}_E) = X_{-E}^T(y-X_E\beta_E)-X_{-E}^TX_E(\bar{\beta}_E-\beta_E)
                        = X_{-E}^T(y-X_E\beta_E)-X_{-E}^TX_{E}(X_E^TX_E)^{-1}X_E^T(y-X_E\beta_E)
                      bootstrapped version becomes Z_star[inactive, ]-(X^{*T}_{-E} X^*_E)(X_E^{*T} X_E^*)^{-1} Z_star[active, ]
        self._cov_beta_bar : (X_E^{*T}X_E^*)^{-1}X_E^{*T}(y^*-X_E^{*}\bar{\beta}_E)
        """

        if not hasattr(self, "_beta_unpenalized"):
            raise ValueError("method fit_E has to be called before computing the covariance")

        if not hasattr(self, "_cov_XTepsilon"):

            # nonparametric bootstrap for covariance of X^T\epsilon

            X, y = self.X, self.y
            n, p = X.shape
            nsample = 5000

            active = self.active
            inactive = ~active

            # computes X_E\bar{\beta}_E
            def pi(X):
                 return np.dot(X[:,self.active], self._beta_unpenalized)
            #    w = np.exp(np.dot(X[:,self.active], self._beta_unpenalized))
            #    return w / (1 + w)

            _mean_cum_Z = 0
            # The following will become the bootstrapped covariance of X^T\epsilon.
            self._cov_XTepsilon = np.zeros((p, p))
            # The following will become the bootstrapped covariance of X_E^T\epsilon.
            self._cov_XETepsilon = np.zeros((self.size_active, self.size_active))

            _mean_cum_N = 0
            # The following will become the bootstrapped covariance of the null statistic X_{-E}^T(y-X_E\bar{\beta}_E).
            self._cov_N = np.zeros((p-self.size_active, p-self.size_active))

            _mean_cum_beta_bar = 0
            # bootstrapped covariance of \bar{\beta}_E, (X_E^{*T}X_E^*)^{-1}X_E^{*T}(y^*-X_E^*\bar{\beta}_E)
            self._cov_beta_bar = np.zeros((self.size_active, self.size_active))

            self._XTX_b = np.zeros((p,p))


            for _ in range(nsample):
                indices = np.random.choice(n, size=(n,), replace=True)
                y_star = y[indices]
                X_star = X[indices]

                self._XTX_b += np.dot(X_star.T, X_star)

                #Z_star = np.dot(X_star.T, y_star - pi(X_star))  # X^{*T}(y^*-X^{*T}_E\bar{\beta}_E)
                Z_star = np.dot(X_star.T, y_star - np.dot(X_star[:, self.active], self._beta_unpenalized))

                _mean_cum_Z += Z_star
                self._cov_XTepsilon += np.multiply.outer(Z_star, Z_star)

                mat_XEstar = np.linalg.inv(np.dot(X_star[:,active].T, X_star[:,active]))  # (X^{*T}_E X^*_E)^{-1}
                mat_star = np.dot(np.dot(X_star[:, inactive].T, X_star[:,active]), mat_XEstar)
                N_star = Z_star[inactive, ]-np.dot(mat_star, Z_star[active, ])
                _mean_cum_N += N_star
                self._cov_N += np.multiply.outer(N_star, N_star)

                beta_star =  np.dot(mat_XEstar, Z_star[active,])
                #beta_star
                #np.linalg.lstsq(X_star[:, self.active], y_star)[0]-self._beta_unpenalized
                _mean_cum_beta_bar += beta_star
                self._cov_beta_bar += np.multiply.outer(beta_star, beta_star)

            self._XTX_b /= nsample

            self._cov_XTepsilon /= nsample
            _mean_Z = _mean_cum_Z / nsample
            #print 'mean Z', _mean_Z
            self._cov_XTepsilon -= np.multiply.outer(_mean_Z, _mean_Z)

            self._cov_N /= nsample
            _mean_N =_mean_cum_N / nsample
            self._cov_N -= np.multiply.outer(_mean_N, _mean_N)
            self._inv_cov_N = np.linalg.inv(self._cov_N)

            # to get _cov_XETepsilon we need to get [active,:]\times[:,active] block of _cov_XTepsilon
            # mat = self._cov_XTepsilon[self.active,:]
            self._cov_XETepsilon = self._cov_XTepsilon[self.active][:,self.active]


            self._cov_beta_bar /= nsample
            _mean_beta_star = _mean_cum_beta_bar/nsample
            #print 'mean beta', _mean_beta_star
            self._cov_beta_bar -= np.multiply.outer(_mean_beta_star, _mean_beta_star)
            self._inv_cov_beta_bar = np.linalg.inv(self._cov_beta_bar)

            #self.L = np.linalg.cholesky(self._cov) #don't know what this is for


    @property
    def covariance(self, doc = "Covariance of sufficient statistic $X^Ty$."): ## this function not used
        if not hasattr(self, "_cov"):
            print 'bootstrap'
            self.bootstrap_covariance()

        return self._cov


    def gradient(self, data, beta):
        """
        Gradient of the negative log-likelihood, written in terms of the 'data' vector and a parameter \beta.
        Taylor series: gradient=\grad_{\beta} l(\beta; X,y)
        =\grad_{\beta} l(\bar{\beta}_E;X,y)+\hessian_{\beta} l(\bar{\beta}_E; X,Y) (\beta-\bar{\beta}_E)
        l(\beta; X,y)=\frac{1}{2}\|y-X\beta\|^2_2 negative log-likelihood,
        gradient = -(0, N)+X^TX_E(\beta-\bar{\beta}_E), data vector is (\bar{\beta}_E, N)
        data0 = \bar{\beta}_E, data1=(0, N) below
        recall N = X_{-E}^T(y-X_E\bar{\beta}_E), the null statistic
        """

        if not hasattr(self, "_cov"):
            self.bootstrap_covariance()

        # g = -(data - np.dot(self._cov, beta))

        data1 = data.copy()

        #data0 = data1[range(self.size_active)].copy()  # \bar{beta}_E, the first |E| coordinates of 'data' vector

        data1[:self.size_active] = 0 # last p-|E| coordinates of data vector kept, first |E| become zeros
                                             # (0, N), N is the null statistic, N=X_{-E}^Y(y-X_E\bar{\beta}_E)
        # print data1

        # g = - data1 + np.dot(self._XTXE, beta[self.active]-data[:self.size_active])

        #g = np.dot(self._XTXE, beta[self.active]-data[:self.size_active])

        g = - data1 + np.dot(self._XTX_b[:, self.active], beta[self.active]-data[:self.size_active])

        return g


    def hessian(self):#, data, beta):
        """
        hessian is constant in this case.
        """
        #if not hasattr(self, "_XTX_b"):
        #    self.bootstrap_covariance()

        return self._XTX


    def setup_sampling(self, data, beta, linear_part, value):
        """
        Set up the sampling conditioning on the linear constraints L * data = value

        Parameters:
        ----------
        data:

        The subject of the sampling. 'data' vector set in tests/test_lasso_randomX.py

        mean: \beta^0_E # ?

        sigma: default to None in logistic lasso # ?

        linear_part: L
        value: fixed value on which we condition L*data=value
        """
        self.L = linear_part

        self.accept_data = 0
        self.total_data = 0

        # projection on the column space of L^T, P = L^T(LL^T)^{-1}L
        # LP=L, const=L*data=L(P*data+(I-P)*data)=LP*data+L(I-P)*data
        P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)  #pinv(L)=L^T(LL^T)^{-1}

        I = np.identity(linear_part.shape[1])

        self.data = data
        self.beta = beta[self.active]

        # print 'beta', self.beta
        # L(I-P)=LR=0, hence for new_data = data+R*proposal, we have L*new_data = L*data+LR*proposal = L*data=constant
        self.R = I - P

        self.P = P

        self.linear_part = linear_part


    def proposal(self, data): #, val):
        # if not hasattr(self, "L"):  # don't know what this is for
        #    self.bootstrap_covariance()

        n, p = self.X.shape
        stepsize = 15. / np.sqrt(p)   # 20 for the selected model

        #stepsize=15./p

        # the new data point proposed will change the current one only along the direction
        # perpendicular to the column space of L^T (or the residual leftover after projection onto the
        # column space of L^T)

        #new = data + stepsize * np.dot(self.R,
        #                               np.random.standard_normal(p))


        ## bootstrap
        active = self.active
        inactive = ~active
        size_active = self.size_active
        size_inactive = data.shape[0] - size_active

        eta = 0.3
        indices = np.random.choice(n, size=(n,), replace = True)
        indices1 = [i if np.random.uniform(0, 1, 1) < eta else indices[i] for i in range(n)]

        y_star = self.y[indices1]
        X_star = self.X[indices1]
        X_star_E = X_star[:,active]

        mat_XEstar = np.linalg.inv(np.dot(X_star_E.T, X_star_E))  # (X^{*T}_E X^*_E)^{-1}
        Z_star = np.dot(X_star_E.T, y_star - np.dot(X_star_E, self._beta_unpenalized))  # X^{*T}_E(y^*-X^{*T}_E\bar{\beta}_E)

        # selected, additionally bootstrap N
        # Z_star = np.dot(X_star.T, y_star - np.dot(X_star[:,self.active], data[:size_active]))  # X^{*T}(y^*-X^{*T}_E\bar{\beta}_E)

        # mat_XEstar = np.linalg.inv(np.dot(X_star[:,active].T, X_star[:,active]))  # (X^{*T}_E X^*_E)^{-1}
        # mat_star = np.dot(np.dot(X_star[:, inactive].T, X_star[:,active]), mat_XEstar)
        # N_star = Z_star[inactive, ]-np.dot(mat_star, Z_star[active, ])

        # data_star = np.concatenate((Z_star,
        #                           N_star-data[-size_inactive:]), axis=0)

        # saturated

        data_star = np.concatenate((np.dot(mat_XEstar,Z_star), np.zeros(size_inactive)), axis=0)

        # data_star = np.concatenate((np.dot(mat_XEstar,Z_star), data[:size_inactive]), axis=0)

        # new = data + stepsize * np.dot(self.R, data_star)

        new = np.dot(self.P, data) + np.dot(self.R, data_star)

        log_transition_p = self.logpdf(new) - self.logpdf(data)

        return new, log_transition_p



    def logpdf(self, data):

        #sampling_data = np.dot(self.R, data)

        # mat = self.R[self.active,:]
        # R_cut = mat[:,self.active]
        #print 'R_cut size', R_cut.shape

        beta_unpen = data[:self.size_active,]

        # beta_unpen = sampling_data[range(self.size_active)]
        # cov_beta_unpen = np.dot(np.dot(R_cut, self._cov_beta_bar),R_cut.T)
        # inv_cov_beta_unpen = np.linalg.inv(cov_beta_unpen)

        # Z = sampling_data[(self.size_active):data.shape[0]]


        #N = data[(self.size_active):data.shape[0]]

        logl_beta_unpen = - np.dot(np.dot(beta_unpen.T, self._inv_cov_beta_bar), beta_unpen)

        #logl_beta_unpen = - np.dot(np.dot(beta_unpen.T, inv_cov_beta_unpen), beta_unpen)
        #print 'cov N size', self._covN.shape
        #print 'length Z', Z.shape
        #logl_N = - np.dot(np.dot(N.T, self._inv_cov_N), N)
        return logl_beta_unpen #+ logl_N


        #return - np.dot(np.dot((beta_unpen-self.beta).T, self._inv_cov_beta_bar), beta_unpen-self.beta)

        #return -((data-self.mean)*np.dot(np.linalg.pinv(self._cov), data-self.mean)).sum() / 2



    def update_proposal(self, state, proposal, logpdf):
        pass








