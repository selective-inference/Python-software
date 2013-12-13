import numpy as np

from selection.variance_estimation import draw_truncated
from selection.lasso import lasso
from selection.constraints import constraints
 
n, p, s, sigma = 100, 200, 10, 5

X = np.random.standard_normal((n,p)) + np.random.standard_normal(n)[:,None]
X -= X.mean(0)[None,:]
X /= X.std(0)[None,:]
X /= np.sqrt(n)

beta = np.zeros(p)
beta[:s] = 1.5 * np.sqrt(2 * np.log(p)) * sigma
y = np.random.standard_normal(n) * sigma
L = lasso(y, X, frac=0.5)
L.fit(tol=1.e-14, min_its=200)
C = L.inactive_constraints

PR = np.identity(n) - L.PA
try:
    U, D, V = np.linalg.svd(PR)
except np.linalg.LinAlgError:
    D, U = np.linalg.eigh(PR)

keep = D >= 0.5
U = U[:,keep]
Z = np.dot(U.T, y)
Z_inequality = np.dot(C.inequality, U)
Z_constraint = constraints((Z_inequality, C.inequality_offset), None)

trunc_sample = draw_truncated(Z, Z_constraint)
