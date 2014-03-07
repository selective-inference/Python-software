"""
fused_lasso.py
Date: 2014-02-12
Author: Xiaoying Tian
"""

from __future__ import division
import numpy as np
import scipy.io
from selection.intervals import interval
from selection.constraints import constraints
from sklearn.linear_model import Lasso 
import matplotlib.pyplot as plt


def slope_constraints(X, y, lam):
    n, p = X.shape
    Lambda = np.zeros((p,p))
    # solve the slope problem
    lasso_fit = Lasso(alpha=lam, fit_intercept=False)
    lasso_fit.fit(X,y)
    beta = lasso_fit.coef_
    score = np.dot(X.T, y - np.dot(X, beta)) 
    index = sorted(range(p), key=lambda i:abs(score[i]))
    for i in range(p):
        Lambda[index[i], index[i]] = lam
    z = np.linalg.solve(Lambda, score)
    # E = np.where((np.abs(beta)>1e-9) | (z<-1+1e-9) | (z>1-1e-9))[0]
    E = np.where(np.abs(beta)>1e-9)[0]
    Q_e, R_e = np.linalg.qr(X[:,E])
    Proj = np.dot(Q_e, Q_e.T)
    z1 = np.sign(beta[E])
    psinv_e = np.linalg.pinv(X[:,E])
    w = np.dot(psinv_e, np.dot(Proj, y)) 
    X_e = np.dot(Proj, psinv_e.T)
    A = np.diag(z1)
    b = -np.diag(z1).dot(psinv_e).dot(psinv_e.T).dot(Lambda[E,E]).dot(z1)
    return [A, b, X_e, beta, E, Proj]

def main():
    # import the CGH data
    mat = scipy.io.loadmat('y.mat')
    v = mat['v']
    v = v.ravel()
    y = v - np.mean(v) 
    # initialize the design matrix 
    n = len(y) 
    D = (np.identity(n) - np.diag(np.ones(n-1),1))[:-1]
    X = np.linalg.pinv(D)
    # get the constraint matrix and vector to compute conf_intervals
    A, b, X_e, beta, E, Proj = slope_constraints(X, y, 0.001)
    C = constraints((A.dot(X_e.T),b), None, covariance=0.007 * np.eye(n))
    # beta_interval = []
    mean_interval = []
    jump = []
    '''
    for i in range(len(E)):
        beta_intv = C.interval(X_e[:,i], y) 
        beta_interval.append(beta_intv)
    '''
    for i in range(len(E)):
        if i == 0 or E[i] - E[i-1] > 5: 
            mean_intv = C.interval(Proj[:,E[i]], y)
            mean_interval.append(mean_intv)
            jump.append(E[i])
        else:
            eta = np.mean(Proj[:,E[i]:(E[i+1]+1)], axis=1)
            mean_intv = C.interval(eta, y)
            mean_interval[-1] = mean_intv
            jump[-1] = E[i]
    mean_interval.append(C.interval(Proj[:,-1], y))
    print mean_interval
    print jump 
    print E
    fit_line = np.dot(X, beta) + np.mean(v)
    low = [l for l,h in mean_interval]
    high = [h for l,h in mean_interval]
    low_fit = np.zeros((n,1))
    high_fit = np.zeros((n,1))
    left = 0
    for idx, e in enumerate(jump):
        right = e
        low_fit[left:right] = low[idx] 
        high_fit[left:right] = high[idx] 
        left = right
    low_fit[left:] = low[-1]
    high_fit[left:] = high[-1]
    low_fit += np.mean(v)
    high_fit += np.mean(v)
    print "before plotting"
    plt.figure()
    plt.scatter(range(n), v, s=40, facecolors='none', edgecolors='r')
    plt.plot(range(n), fit_line, linewidth=2)
    plt.plot(range(n), low_fit, linewidth=2, color='g')
    plt.plot(range(n), high_fit, linewidth=2, color='y')
    plt.xlim(0,n)
    plt.show()



if __name__ == "__main__":
    main()
