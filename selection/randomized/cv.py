import functools
import numpy as np
import regreg.api as rr



def CV_err(loss, penalty, folds, scale=0.5, solve_args={'min_its':20, 'tol':1.e-10}):

    X, y = loss.data
    n, p = X.shape

    CV_err = 0
    CV_err_randomized = 0

    CV_err_squared = 0
    CV_err_squared_randomized = 0

    for fold in np.unique(folds):
        test = folds == fold
        train = ~test

        loss_train = loss.subsample(train)
        loss_test = loss.subsample(test)
        X_test, y_test = loss_test.data
        n_test = y_test.shape[0]
        problem = rr.simple_problem(loss_train, penalty)
        beta_train = problem.solve(**solve_args)

        _mu = lambda X, beta: loss_test.saturated_loss.mean_function(X.dot(beta))
        resid = y_test - _mu(X_test, beta_train)
        cur = (resid**2).sum() / n_test

        # there are several ways we could randomize here...
        random_noise = scale * np.random.standard_normal(y_test.shape)
        cur_randomized = ((resid + random_noise)**2).sum() / n_test

        CV_err += cur
        CV_err_squared += cur**2

        CV_err_randomized += cur_randomized
        CV_err_squared_randomized += cur_randomized**2

    K = len(np.unique(folds))

    SD_CV = np.sqrt((CV_err_squared.mean() - CV_err.mean()**2) / (K-1))
    SD_CV_randomized = np.sqrt((CV_err_squared_randomized.mean() - CV_err_randomized.mean()**2) / (K-1))
    return CV_err, SD_CV, CV_err_randomized, SD_CV_randomized


def choose_lambda_CV(loss, lam_seq, folds):

    CV_curve = []
    X, _ = loss.data
    p = X.shape[1]
    for lam in lam_seq:
        penalty = rr.l1norm(p, lagrange=lam)
        CV_curve.append(CV_err(loss, penalty, folds) + (lam,))

    CV_curve = np.array(CV_curve)
    minCV = lam_seq[np.argmin(CV_curve[:,0])] # unrandomized
    minCV_randomized = lam_seq[np.argmin(CV_curve[:,2])] # randomized

    return minCV_randomized, CV_curve



def bootstrap_CV_curve(loss, lam_seq, folds, K):

    def _bootstrap_CVerr_curve(loss, lam_seq, K, indices):
        X, _ = loss.data
        n, p = X.shape
        folds_star = np.arange(n) % K
        np.random.shuffle(folds_star)
        loss_star = loss.subsample(indices)
        return np.array(choose_lambda_CV(loss_star, lam_seq, folds_star)[1])[:,0]

    return functools.partial(_bootstrap_CVerr_curve, loss, lam_seq, K)

