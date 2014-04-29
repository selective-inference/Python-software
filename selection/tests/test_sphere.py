# IPython log file

import numpy as np
from selection.forward_step import forward_stepwise
from selection.covtest import covtest
from selection.affine import constraints, simulate_from_sphere
n, p, sigma = 20,30,1.5

X = (np.random.standard_normal((n,p)) + 
     0.7 * np.random.standard_normal(n)[:,None])
X -= X.mean(0)[None,:]
X /= X.std(0)[None,:] * np.sqrt(n)
beta = np.zeros(p)
Y = np.random.standard_normal(n) * sigma + np.dot(X, beta)

def get_condition(X, Y, sigma=None,
                 nstep=3,
                 tests=['reduced_unknown'],
                 burnin=1000,
                 ndraw=5000):
    """
    A simple implementation of forward stepwise
    that uses the `reduced_covtest` iteratively
    after adjusting fully for the selected variable.

    This implementation is not efficient, in
    that it computes more SVDs than it really has to.

    Parameters
    ----------

    X : np.float((n,p))

    Y : np.float(n)

    nstep : int
        How many steps of forward stepwise?

    sigma : float (optional) 
        Noise level (not needed for reduced).

    tests : ['reduced_known', 'covtest', 'reduced_unknown']
        Which test to use? A subset of the above sequence.

    """

    n, p = X.shape
    FS = forward_stepwise(X, Y)

    for i in range(nstep):
        FS.next()

        Acon = constraints(FS.A, np.zeros(FS.A.shape[0]))
        if i > 0:
            U = FS.P[-2].U.T
            Uy = np.dot(U, Y)
            Acon = Acon.conditional(U, Uy)
        else:
            Acon = Acon

    return Acon, Y

con, Y = get_condition(X, Y)
I, F, conW = con.whiten()
A = conW.linear_part
b = conW.offset

from selection.affine import sample_truncnorm_white_sphere

while True:
    w = np.random.standard_normal(A.shape[1])
    if (np.dot(A, w) - b).max() < 0:
        break

Z = simulate_from_sphere(con, Y)


# sample_truncnorm_white_sphere(A, b, w, burnin=1000, ndraw=2000)
