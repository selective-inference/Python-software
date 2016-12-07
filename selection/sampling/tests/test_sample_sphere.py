from __future__ import absolute_import, print_function
import nose
import nose.tools as nt
import numpy as np
import numpy.testing.decorators as dec

from scipy.stats import chi
import nose.tools as nt


from selection.tests.flags import SET_SEED, SMALL_SAMPLES
import selection.constraints.affine as AC
from selection.algorithms.lasso import lasso
from selection.algorithms.sqrt_lasso import choose_lambda
from selection.distributions.discrete_family import discrete_family
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

# generate a cone from a sqrt_lasso problem

def _generate_constraints(n=15, p=10, sigma=1):
    while True:
        y = np.random.standard_normal(n) * sigma
        beta = np.zeros(p)
        X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
        X /= (X.std(0)[None,:] * np.sqrt(n))
        y += np.dot(X, beta) * sigma
        lam_theor = 0.3 * choose_lambda(X, quantile=0.9)
        L = lasso.sqrt_lasso(X, y, lam_theor)
        L.fit(solve_args={'tol':1.e-12, 'min_its':150})

        con = L.constraints
        if con is not None and L.active.shape[0] >= 3:
            break

    offset = con.offset
    linear_part = -L.active_signs[:,None] * np.linalg.pinv(X[:,L.active])
    con = AC.constraints(linear_part, offset)
    con.covariance = np.identity(con.covariance.shape[0])
    con.mean *= 0
    return con, y, L, X

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_sample_ball(burnin=1000,
                     ndraw=1000):

    p = 10
    A = np.identity(10)[:3]
    b = np.ones(3)
    initial = np.zeros(p)
    eta = np.ones(p)

    bound = 5
    s = AC.sample_truncnorm_white_ball(A,
                                       b, 
                                       initial,
                                       eta,
                                       lambda state: bound + 0.01 * np.random.sample() * np.linalg.norm(state)**2,
                                       burnin=burnin,
                                       ndraw=ndraw,
                                       how_often=5)
    return s

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
def test_sample_sphere(burnin=1000,
                       ndraw=1000):

    p = 10
    A = np.identity(10)[:3]
    b = 2 * np.ones(3)
    mean = -np.ones(p)
    noise = np.random.standard_normal(p) * 0.1
    noise[-3:] = 0.
    initial = noise + mean
    eta = np.ones(p)

    bound = 5
    s1 = AC.sample_truncnorm_white_sphere(A,
                                          b, 
                                          initial,
                                          eta,
                                          how_often=20,
                                          burnin=burnin,
                                          ndraw=ndraw)

    con = AC.constraints(A, b)
    con.covariance = np.diag([1]*7 + [0]*3)
    con.mean[:] = mean
    s2 = AC.sample_from_sphere(con, initial, ndraw=ndraw, burnin=burnin)
    return s1, s2

@dec.slow
@set_seed_iftrue(SET_SEED, 20)
@set_sampling_params_iftrue(SMALL_SAMPLES, nsim=10, ndraw=10, burnin=10)
def test_distribution_sphere(n=15, p=10, sigma=1.,
                             nsim=2000,
                             sample_constraints=False,
                             burnin=10000,
                             ndraw=10000):

    # see if we really are sampling from 
    # correct distribution
    # by comparing to an accept-reject sampler

    con, y = _generate_constraints()[:2]
    accept_reject_sample = []

    hit_and_run_sample, W = AC.sample_from_sphere(con, y, 
                                                  ndraw=ndraw,
                                                  burnin=burnin)
    statistic = lambda x: np.fabs(x).max()
    family = discrete_family([statistic(s) for s in hit_and_run_sample], W)
    radius = np.linalg.norm(y)

    count = 0

    pvalues = []

    while True:

        U = np.random.standard_normal(n)
        U /= np.linalg.norm(U)
        U *= radius

        if con(U):
            accept_reject_sample.append(U)
            count += 1

            true_sample = np.array([statistic(s) for s in accept_reject_sample])

            if (count + 1) % int(nsim / 10) == 0:

                pvalues.extend([family.cdf(0, t) for t in true_sample])
                print(np.mean(pvalues), np.std(pvalues))

                if sample_constraints:
                    con, y = _generate_constraints()[:2]

                hit_and_run_sample, W = AC.sample_from_sphere(con, y, 
                                                              ndraw=ndraw,
                                                              burnin=burnin)
                family = discrete_family([statistic(s) for s in hit_and_run_sample], W)
                radius = np.linalg.norm(y)
                accept_reject_sample = []

        if count >= nsim:
            break

    U = np.linspace(0, 1, 101)

#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm 

#     plt.plot(U, sm.distributions.ECDF(pvalues)(U))
#     plt.plot([0,1],[0,1])

@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
def test_conditional_sampling(n=20, p=25, sigma=20,
                              ndraw=1000,
                              burnin=1000):
    """
    goodness of fit samples from
    inactive constraints intersect a sphere

    this test verifies the sampler is doing what it should
    """

    con, y, L, X = _generate_constraints(n=n, p=p, sigma=sigma)

    X_E = X[:,L.active]
    C_Ei = np.linalg.pinv(X_E.T.dot(X_E))
    R_E = lambda z: z - X_E.dot(C_Ei.dot(X_E.T.dot(z)))

    X_minus_E = X[:,L.inactive]
    RX_minus_E = R_E(X_minus_E)
    inactive_bound = L.feature_weights[L.inactive]
    active_subgrad = L.feature_weights[L.active] * L.active_signs
    irrep_term = X_minus_E.T.dot(X_E.dot(C_Ei.dot(active_subgrad)))

    inactive_constraints = AC.constraints(
                             np.vstack([RX_minus_E.T,
                                        -RX_minus_E.T]),
                             np.hstack([inactive_bound - irrep_term,
                                        inactive_bound + irrep_term]),
                             covariance = np.identity(n)) 

    con = inactive_constraints
    conditional_con = con.conditional(X_E.T, np.dot(X_E.T, y))

    Z, W = AC.sample_from_sphere(conditional_con, 
                                 y,
                                 ndraw=ndraw,
                                 burnin=burnin)  
    
    T1 = np.dot(X_E.T, Z.T) - np.dot(X_E.T, y)[:,None]
    nt.assert_true(np.linalg.norm(T1) < 1.e-7)

    T2 = (R_E(Z.T)**2).sum(0) - np.linalg.norm(R_E(y))**2
    nt.assert_true(np.linalg.norm(T2) < 1.e-7)
