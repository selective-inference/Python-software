import functools
import numpy as np
import regreg.api as rr
import copy
from selection.randomized.M_estimator import restricted_Mest

class CV(object):

    def __init__(self, loss, folds, lam_seq, objective_randomization=None, epsilon=None):

        (self.loss,
         self.folds,
         self.lam_seq,
         self.objective_randomization,
         self.epsilon) = (loss,
                          folds,
                          lam_seq,
                          objective_randomization,
                          epsilon)

        if self.objective_randomization is not None:
            if self.epsilon is None:
                X, _ = loss.data
                n = X.shape[0]
                self.epsilon = np.true_divide(1, np.sqrt(n))
        self.K = len(np.unique(self.folds))

    def CV_err(self,
               penalty,
               loss = None,
               residual_randomization = None,
               scale = None,
               solve_args={'min_its':20, 'tol':1.e-1}):
        """
        Computes the non-randomized CV error and the one with added residual randomization
        """
        if loss is None:
            loss = copy.copy(self.loss)
        X, y = loss.data
        n, p = X.shape

        CV_err = 0
        CV_err_squared = 0

        if residual_randomization is not None:
            CV_err_randomized = 0
            CV_err_squared_randomized = 0
            if scale is None:
                scale = 1.

        for fold in np.unique(self.folds):
            test = self.folds == fold
            train = ~test

            loss_train = loss.subsample(train)
            loss_test = loss.subsample(test)
            X_test, y_test = X[test], y[test]
            n_test = y_test.shape[0]

            if self.objective_randomization is not None:
                randomized_train_loss = self.objective_randomization.randomize(loss_train, self.epsilon) # randomized train loss
                problem = rr.simple_problem(randomized_train_loss, penalty)
            else:
                problem = rr.simple_problem(loss_train, penalty)
            beta_train = problem.solve(**solve_args)

            #active = beta_train!=0
            #_beta_unpenalized = restricted_Mest(loss_train, active, solve_args=solve_args)
            #beta_full = np.zeros(p)
            #beta_full[active] = _beta_unpenalized

            _mu = lambda X, beta: loss_test.saturated_loss.mean_function(X.dot(beta))
            resid = y_test - _mu(X_test, beta_train)
            cur = (resid**2).sum() / n_test
            CV_err += cur
            CV_err_squared += (cur**2)

            if residual_randomization is not None:
                random_noise = scale * np.random.standard_normal(n_test)
                cur_randomized = ((resid + random_noise)**2).sum() / n_test
                CV_err_randomized += cur_randomized
                CV_err_squared_randomized += cur_randomized**2

        SD_CV = np.sqrt((CV_err_squared - ((CV_err**2)/self.K)) / float(self.K-1))
        if residual_randomization is not None:
            SD_CV_randomized = np.sqrt((CV_err_squared_randomized - (CV_err_randomized**2/self.K)) / (self.K-1))
            return CV_err, SD_CV, CV_err_randomized, SD_CV_randomized
        else:
            #print(CV_err, SD_CV)
            return CV_err, SD_CV


    def choose_lambda_CVr(self, scale = 1., loss=None):
        """
        Minimizes CV error curve without randomization and the one with residual randomization
        """
        if loss is None:
            loss = self.loss

        if not hasattr(self, 'scale'):
            self.scale = scale

        CV_curve = []
        X, _ = loss.data
        p = X.shape[1]
        for lam in self.lam_seq:
            penalty = rr.l1norm(p, lagrange=lam)
            # CV_curve.append(self.CV_err(penalty, loss) + (lam,))
            CV_curve.append(self.CV_err(penalty, loss, residual_randomization = True, scale = self.scale))

        CV_curve = np.array(CV_curve)
        CV_val = CV_curve[:,0]
        CV_val_randomized = CV_curve[:,2]
        lam_CV = self.lam_seq[np.argmin(CV_val)]
        lam_CV_randomized = self.lam_seq[np.argmin(CV_val_randomized)]

        return lam_CV, CV_val, lam_CV_randomized, CV_val_randomized

    def bootstrap_CVr_curve(self):
        """
        Bootstrap of CV error curve with residual randomization
        """
        def _boot_CVr_curve(indices):
            X, y = self.loss.data
            n, p = X.shape
            folds_star = np.arange(n) % self.K
            np.random.shuffle(folds_star)
            loss_star = self.loss.subsample(indices)
            #loss_star = rr.glm.gaussian(X[indices,:], y[indices])
            _, _, _, CV_val_randomized = self.choose_lambda_CVr(scale=self.scale, loss=loss_star)
            return np.array(CV_val_randomized)

        return _boot_CVr_curve


    def choose_lambda_CVR(self,  randomization1=None, randomization2=None, loss=None):
        """
        Minimizes CV error curve with additive randomization (CVR=CV+R1+R2=CV1+R2)
        """
        if loss is None:
            loss = copy.copy(self.loss)
        CV_curve = []
        X, _ = loss.data
        p = X.shape[1]
        for lam in self.lam_seq:
            penalty = rr.l1norm(p, lagrange=lam)
            #CV_curve.append(self.CV_err(penalty, loss) + (lam,))
            CV_curve.append(self.CV_err(penalty, loss))

        CV_curve = np.array(CV_curve)

        rv1, rv2 = np.zeros(self.lam_seq.shape[0]), np.zeros(self.lam_seq.shape[0])
        if randomization1 is not None:
            rv1 = np.asarray(randomization1._sampler(size=(1,)))
        if randomization2 is not None:
            rv2 = np.asarray(randomization2._sampler(size=(1,)))
        CVR_val = CV_curve[:,0]+rv1.flatten()+rv2.flatten()
        lam_CVR = self.lam_seq[np.argmin(CVR_val)] # lam_CVR minimizes CVR
        CV1_val = CV_curve[:,0]+rv1.flatten()

        SD = CV_curve[:,1]
        return lam_CVR, SD, CVR_val, CV1_val


    def bootstrap_CVR_curve(self, randomization1=None, randomization2=None):
        """
        Bootstrap of CVR=CV+R1+R2 and CV1=CV+R1 curves
        """
        def _bootstrap_CVerr_curve(indices):
            X, y = self.loss.data
            n, p = X.shape
            folds_star = np.arange(n) % self.K
            np.random.shuffle(folds_star)
            loss_star = self.loss.subsample(indices)
            #loss_star = rr.glm.gaussian(X[indices,:], y[indices])
            _, _, CVR_val, CV1_val = self.choose_lambda_CVR(randomization1, randomization2, loss_star)
            return np.array(CVR_val), np.array(CV1_val)

        def _CVR_boot(indices):
            return _bootstrap_CVerr_curve(indices)[0]
        def _CV1_boot(indices):
            return _bootstrap_CVerr_curve(indices)[1]

        return _CVR_boot, _CV1_boot
