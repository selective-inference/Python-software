import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.bayesian.objective_functions import my_selection_probability_only_objective, \
    selection_probability_only_objective, dual_selection_probability_only_objective
from selection.bayesian.selection_probability import selection_probability_methods

#fixing n, p, true sparsity and signal strength
n=20
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

    B_sel_1 = np.zeros((p, p))
    B_sel_1[:, :nactive] = np.dot(X_perm.T, X_perm[:, :nactive])* active_signs[None, :]
    B_sel_1[:nactive, :nactive] += (epsilon * np.identity(nactive))* active_signs[None, :]
    B_sel_1[nactive:, nactive:] = np.identity((p - nactive))

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
        parameter = np.fabs(np.random.standard_normal(nactive))
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

    #print test_objectives_compare()

    def test_objectives_compare_0():
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        vec = np.random.standard_normal(n)
        active_coef = np.dot(np.diag(active_signs),np.fabs(np.random.standard_normal(nactive)))
        mean = X_1[:, active].dot(parameter)
        sel = my_selection_probability_only_objective(V, B_sel, gamma_sel, noise_variance, tau, lam, y, betaE, cube, vec,
                                                      active_coef)
        sel_breakup = sel.optimization(parameter)[0] + np.true_divide(np.dot(mean.T, mean), 2 * noise_variance) + \
                      np.true_divide(np.dot(gamma_sel[:nactive].T, gamma_sel[:nactive]), 2 * (tau ** 2)), \
                      sel.optimization(parameter)[1]
        sel_prob = selection_probability_only_objective(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                   noise_variance, randomization.isotropic_gaussian((p,), 1.),
                                                   epsilon)
        sel_prob_breakup = sel_prob.smooth_objective(np.append(vec,np.fabs(active_coef)), mode='func', check_feasibility=False)
        return sel_breakup, sel_prob_breakup

    #print test_objectives_compare_0()


    def test_dual():
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        feasible = np.append(-np.fabs(np.random.standard_normal(nactive)), np.random.standard_normal((p-nactive)))
        dual_feasible = np.dot(np.linalg.inv(B_sel_1.T), feasible)
        #check = np.dot(B_sel_1.T, dual_feasible)
        sel_prob = dual_selection_probability_only_objective(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                   noise_variance, randomization.isotropic_gaussian((p,), 1.),
                                                   epsilon)
        sel_prob_obj = sel_prob.objective(dual_feasible)
        #deviation = mean - np.dot(X_1, dual_feasible)
        #check = (dual_feasible.T.dot(dual_feasible)/2), (deviation.T.dot(deviation)/2), dual_feasible.T.dot(gamma_sel)
        sel_prob_min = sel_prob.opt_minimize()[0]- np.true_divide(np.dot(mean.T, mean), 2 * noise_variance)
        #check = sel_prob.objective(dual_feasible)
        return sel_prob_min


    #print test_dual()


    def test_new_function():
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        vec = np.random.standard_normal(n)
        active_coef = np.dot(np.diag(active_signs), np.fabs(np.random.standard_normal(nactive)))
        sel = my_selection_probability_only_objective(V, B_sel, gamma_sel, noise_variance, tau, lam, y, betaE, cube,
                                                      vec,
                                                      active_coef)
        sel_breakup = sel.optimization(parameter)[0] + np.true_divide(np.dot(mean.T, mean), 2 * noise_variance) + \
                      np.true_divide(np.dot(gamma_sel[:nactive].T, gamma_sel[:nactive]), 2 * (tau ** 2)), \
                      sel.optimization(parameter)[1]

        sel_prob = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                   noise_variance, tau, epsilon)

        sel_breakup_1 = sel_prob.objective(np.append(vec,np.fabs(active_coef)))

        print sel_breakup, sel_breakup_1


    #test_new_function()

    def test_objective_new_function():
        if nactive==1:
            snr_seq = np.linspace(-10, 10, num=100)
            num = snr_seq.shape[0]
            lagrange = lam * np.ones(p)
            for i in range(snr_seq.shape[0]):
                parameter = snr_seq[i]
                print "parameter value", parameter
                mean = X_1[:, active].dot(parameter)
                vec = np.random.standard_normal(n)
                active_coef = np.dot(np.diag(active_signs), np.fabs(np.random.standard_normal(nactive)))
                sel = my_selection_probability_only_objective(V, B_sel, gamma_sel, noise_variance, tau, lam, y, betaE,
                                                              cube,
                                                              vec,
                                                              active_coef)
                sel_breakup = sel.optimization(parameter*np.ones(nactive))[0] + np.true_divide(np.dot(mean.T, mean),
                                                                              2 * noise_variance) + \
                              np.true_divide(np.dot(gamma_sel[:nactive].T, gamma_sel[:nactive]), 2 * (tau ** 2)), \
                              sel.optimization(parameter*np.ones(nactive))[1]

                sel_prob = selection_probability_methods(X_1, np.fabs(betaE), active, active_signs, lagrange, mean,
                                                        noise_variance, tau, epsilon)

                sel_breakup_1 = sel_prob.objective(np.append(vec, np.fabs(active_coef)))

                print sel_breakup, sel_breakup_1









