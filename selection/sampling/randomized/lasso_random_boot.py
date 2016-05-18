import numpy as np
from scipy.stats import laplace, probplot, uniform
from copy import copy
from selection.algorithms.lasso import instance, lasso, standard_lasso
import selection.sampling.randomized.api as randomized

from matplotlib import pyplot as plt
import regreg.api as rr
from sklearn import linear_model


def bootstrap(y, X, residuals, active, i, j,  nsample=1000):
    n,p = X.shape
    eta = X[:, j]
    keep = np.copy(active)
    keep[j] = False
    linear_part = X[:, keep].T

    P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
    I = np.identity(linear_part.shape[1])
    R = I - P

    boot_samples = []
    comparison = []
    beta_bar = np.linalg.lstsq(X[:, active],y)[0]
    y_pred = np.dot(X[:,active], beta_bar)
    #print 'beta_bar', beta_bar.shape[0]
    for _ in range(nsample):
        #print i
        indices = np.random.choice(n, size=(n,), replace=True)
        residuals_star = residuals[indices]

        y_star = np.dot(P,y) + np.dot(R, y_pred+residuals_star)

        boot_samples.append(y_star)

        beta_star = np.linalg.lstsq(X[:, active], y_pred+residuals_star)[0]

        #print 'beta_star_size', beta_star.shape[0]
        #print 'beta_star[i],i', i, beta_star[i]
        comparison.append(beta_star[i]>(2*beta_bar[i]))

    return boot_samples, comparison


def randomization_cdf(randomization, boot_samples, A, b):

    rand = randomization
    prob_selection = []
    for _, y_star in enumerate(boot_samples):
        p = rand.cdf(b-np.dot(A, y_star)) ## CORRECT HERE
        #print 'cdf', p, np.prod(p)
        prob_selection.append(np.prod(p))
    return prob_selection


def test_lasso(s=5, n=100, p=10, randomization = laplace(0,1)):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)
    lam_frac = 1.

    lam = sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    #penalty = glm.gaussian(X, Y, coef=1. / sigma**2, quadratic=quadratic)

    #loss =
    #problem = rr.simple_problem(loss, penalty)
    #solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500})
    #initial_soln = problem.solve(**solve_args)
    clf = linear_model.Lasso(alpha = lam/(2*float(n)))
    clf.fit(X,y)
    soln = clf.coef_
    active = (soln !=0)  # boolean vector
    active_set = np.where(active)[0] # column numbers of covariates chosen by lasso
    print 'active', active
    print 'active_set', active_set
    active_size = np.sum(active)
    print 'size of the active set', active_size

    inactive = ~active
    signs = np.sign(soln[active])
    #print signs
    print 'true support', nonzero
    pseudo_X_M = np.linalg.pinv(X[:, active])
    pseudo_XT_M = np.linalg.pinv(X[:, active].T)

    P_M = np.dot(X[:, active], pseudo_X_M)
    #print 'active', X[:, active_set]
    #print np.dot(P_M, X[:, active_set])
    A01 = np.dot(X[:,inactive].T, np.identity(n) -P_M)/lam
    A02 = - A01.copy()
    #print 'A01',A01
    #print 'A02',A02
    A0 = np.concatenate((A01, A02), axis=0)
    #print 'A0', A0

    A1 = - np.dot(np.diag(signs),pseudo_X_M)
    A = np.concatenate((A0,A1), axis=0)
    #print signs
    #print pseudo_X_M
    #print A1
    b01 = np.ones(p-active_size) - np.dot(np.dot(X[:, inactive].T, pseudo_XT_M), signs)
    b02 = np.ones(p-active_size) + np.dot(np.dot(X[:, inactive].T, pseudo_XT_M), signs)
    b0 = np.concatenate((b01,b02), axis=0)
    mat = np.linalg.inv(np.dot(X[:,active].T, X[:, active]))
    b1 = -lam * np.dot(np.dot(np.diag(signs),mat), signs)
    b = np.concatenate((b0,b1), axis=0)

    beta_bar = np.linalg.lstsq(X[:, active],y)[0]
    residuals = y - np.dot(X[:, active], beta_bar)

    null, alt = [],[]

    for i, j in enumerate(active_set):
        boot_samples, comparison = bootstrap(y, X, residuals, active, i, j)
        prob_selection = randomization_cdf(randomization, boot_samples, A, b)
        print 'comparison', np.sum(comparison)
        #print np.asarray(comparison, dtype=int).shape
        num = np.inner(np.asarray(comparison, dtype=int), np.asarray(prob_selection))
        #print 'num', num
        den = np.sum(np.asarray(prob_selection))
        #print 'den', den
        p_value = num/den
        obs = beta_bar[i]
        print "observed: ", obs, "p value: ", p_value
        if j in nonzero:
            print
            alt.append(p_value)
        else:
            null.append(p_value)
    return null, alt


P0, PA = [], []
for i in range(100):
    print "iteration", i
    p0, pA = test_lasso()
    P0.extend(p0)
    PA.extend(pA)

print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=True)
plt.show()
