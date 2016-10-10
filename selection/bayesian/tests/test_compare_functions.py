import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.objective_functions import my_selection_probability_only_objective, selection_probability_only_objective


#fixing n, p, true sparsity and signal strength
n=30
p=3
s=1
snr=5

#sampling the Gaussian instance
X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
random_Z = np.random.standard_normal(p)
#getting randomized Lasso solution
sel = selection(X_1,y, random_Z)

#proceed only if selection is non-empty
if sel is not None:
    lam, epsilon, active, betaE, cube, initial_soln = sel
    print epsilon, lam, betaE
    noise_variance = 1
    nactive=betaE.shape[0]
    active_signs = np.sign(betaE)
    tau=1
    X_perm=np.zeros((n,p))
    X_perm[:,:nactive]=X_1[:,active]
    X_perm[:,nactive:]=X_1[:,~active]
    V=-X_perm
    X_active=X_perm[:,:nactive]
    X_nactive=X_perm[:,nactive:]
    B_sel=np.zeros((p,p))
    B_sel[:,:nactive]=np.dot(X_perm.T,X_perm[:,:nactive])
    B_sel[:nactive, :nactive]+= epsilon*np.identity(nactive)
    B_sel[nactive:, nactive:]=lam*np.identity((p-nactive))
    gamma_sel=np.zeros(p)
    gamma_sel[:nactive]=lam*np.sign(betaE)


    def test_one_sparse_compare():
        if nactive==1:
            snr_seq = np.linspace(-10, 10, num=20)
            lagrange = lam * np.ones(p)
            for i in range(snr_seq.shape[0]):
                parameter = snr_seq[i]*np.ones(nactive)
                print "parameter value", parameter
                mean = X_1[:, active].dot(parameter)
                #the objective of my function--
                sel = my_selection_probability_only_objective(V, B_sel, gamma_sel, noise_variance, tau, lam, y, betaE, cube)
                sel_breakup = sel.optimization(parameter)[0] + np.true_divide(np.dot(mean.T, mean),
                                                                              2 * noise_variance) + \
                              np.true_divide(np.dot(gamma_sel[:nactive].T, gamma_sel[:nactive]), 2 * (tau ** 2)), \
                              sel.optimization(parameter)[1]
                #the objective of Jonathan's function--
                sel_prob = selection_probability_only_objective(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                           noise_variance, randomization.isotropic_gaussian((p,), 1.),
                                                           epsilon)
                sel_prob_breakup = sel_prob.smooth_objective(np.append(y, np.fabs(betaE)), mode='func',
                                                             check_feasibility=False)
                print sel_breakup, sel_prob_breakup


    #test_one_sparse_compare()

    def test_objectives_compare():
        parameter = -np.fabs(np.random.standard_normal(nactive))
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        sel = my_selection_probability_only_objective(V, B_sel, gamma_sel, noise_variance, tau, lam, y, betaE, cube)
        sel_breakup = sel.optimization(parameter)[0] + np.true_divide(np.dot(mean.T, mean), 2 * noise_variance) + \
                      np.true_divide(np.dot(gamma_sel[:nactive].T, gamma_sel[:nactive]), 2 * (tau ** 2)), \
                      sel.optimization(parameter)[1]
        sel_prob = selection_probability_only_objective(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                   noise_variance, randomization.isotropic_gaussian((p,), 1.),
                                                   epsilon)
        sel_prob_breakup = sel_prob.smooth_objective(np.append(y, np.fabs(betaE)), mode='func', check_feasibility=False)
        return sel_breakup, sel_prob_breakup

    print test_objectives_compare()

