import functools
import numpy as np
import regreg.api as rr



def CV_err(loss,
           penalty,
           folds,
           lasso_randomization, epsilon,
           scale=0.5,
           solve_args={'min_its':20, 'tol':1.e-10}):

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
        X_test, y_test = X[test], y[test]
        n_test = y_test.shape[0]

        randomized_train_loss = lasso_randomization.randomize(loss_train, epsilon)

        problem = rr.simple_problem(randomized_train_loss, penalty)
        beta_train = problem.solve(**solve_args)

        _mu = lambda X, beta: loss_test.saturated_loss.mean_function(X.dot(beta))
        resid = y_test - _mu(X_test, beta_train)
        cur = (resid**2).sum() / n_test

        # there are several ways we could randomize here...
        random_noise = scale * np.random.standard_normal(n_test)
        cur_randomized = ((resid + random_noise)**2).sum() / n_test

        CV_err += cur
        CV_err_squared += cur**2

        CV_err_randomized += cur_randomized
        CV_err_squared_randomized += cur_randomized**2

    K = len(np.unique(folds))

    SD_CV = np.sqrt((CV_err_squared.mean() - CV_err.mean()**2) / (K-1))
    SD_CV_randomized = np.sqrt((CV_err_squared_randomized.mean() - CV_err_randomized.mean()**2) / (K-1))
    return CV_err, SD_CV, CV_err_randomized, SD_CV_randomized


def choose_lambda_CV(loss,
                     lam_seq,
                     folds,
                     randomization1, randomization2,
                     lasso_randomization, epsilon):

    CV_curve = []
    X, _ = loss.data
    p = X.shape[1]
    for lam in lam_seq:
        penalty = rr.l1norm(p, lagrange=lam)
        CV_curve.append(CV_err(loss, penalty, folds, lasso_randomization, epsilon) + (lam,))

    CV_curve = np.array(CV_curve)
    #print("nonradomized", CV_curve[:,0])
    minCV = lam_seq[np.argmin(CV_curve[:,0])] # unrandomized
    minCV_randomized = lam_seq[np.argmin(CV_curve[:,2])] # randomized


    rv1 = np.asarray(randomization1._sampler(size=(1,)))
    rv2 = np.asarray(randomization2._sampler(size=(1,)))
    CVR_val = CV_curve[:,0]+rv1.flatten()+rv2.flatten()
    lam_CVR = lam_seq[np.argmin(CVR_val)] # lam_CVR minimizes CVR
    CV1_val = CV_curve[:,0]+rv1.flatten()


    return lam_CVR, CVR_val, CV1_val


def bootstrap_CV_curve(loss,
                       lam_seq,
                       folds,
                       K,
                       randomization1, randomization2,
                       lasso_randomization, epsilon):

    def _bootstrap_CVerr_curve(loss, lam_seq, K, indices):
        X, y = loss.data
        n, p = X.shape
        folds_star = np.arange(n) % K
        np.random.shuffle(folds_star)
        #loss_star = loss.subsample(indices)
        loss_star = rr.glm.gaussian(X[indices,:], y[indices])
        _, CVR_val, CV1_val = choose_lambda_CV(loss_star, lam_seq, folds_star, randomization1, randomization2, lasso_randomization, epsilon)
        return np.array(CVR_val), np.array(CV1_val)

    def _CVR_boot(indices):
        return _bootstrap_CVerr_curve(loss, lam_seq, K, indices)[0]
    def _CV1_boot(indices):
        return _bootstrap_CVerr_curve(loss, lam_seq, K, indices)[1]

    return _CVR_boot, _CV1_boot
    #return functools.partial(_bootstrap_CVerr_curve, loss, lam_seq, K)

