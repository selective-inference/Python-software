import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
from selection.bayesian.dual_scipy import barrier_conjugate_func, dual_selection_probability_func
from selection.bayesian.sel_probability import selection_probability
from selection.bayesian.objective_functions import dual_selection_probability_only_objective
from selection.randomized.api import randomization

n=20
p=5
s=1
snr=5

#sampling the Gaussian instance
X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
random_Z = np.random.standard_normal(p)
#getting randomized Lasso solution
sel = selection(X_1,y, random_Z)

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
    B_sel_1[:, :nactive] = np.dot(X_perm.T, X_perm[:, :nactive]) * active_signs[None, :]
    B_sel_1[:nactive, :nactive] += (epsilon * np.identity(nactive)) * active_signs[None, :]
    B_sel_1[nactive:, nactive:] = np.identity((p - nactive))

    def test_dual():
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        feasible = np.append(-np.fabs(np.random.standard_normal(nactive)), np.random.standard_normal((p-nactive)))
        dual_feasible = np.dot(np.linalg.inv(B_sel_1.T), feasible)
        sel_prob = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                   noise_variance, tau,
                                                   epsilon)
        sel_prob_min = sel_prob.minimize_opt()

        sel = selection_probability(V, B_sel, gamma_sel, noise_variance, tau, lam, y, betaE, cube)
        sel_log_val = sel.optimization(parameter * np.ones(nactive), method="log_barrier")[0] - \
                      np.true_divide(np.dot(mean.T, mean), 2 * noise_variance) \
                      - np.true_divide(np.dot(gamma_sel[:nactive].T, gamma_sel[:nactive]), 2 * (tau ** 2))
        #sel_prob_obj = sel_prob.dual_objective(dual_feasible)
        #return sel_prob.rand_CGF(dual_feasible), sel_prob.composed_barrier_conjugate(dual_feasible),\
        #       sel_prob.data_CGF(dual_feasible),sel_prob.barrier_implicit(dual_feasible),sel_prob.dual_objective(dual_feasible)
        return sel_prob_min[0]- np.true_divide(np.dot(mean.T, mean), 2 * noise_variance), sel_log_val

    #print test_dual()

    def test_minimizer_dual():
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        feasible = np.append(-np.fabs(np.random.standard_normal(nactive)), np.random.standard_normal((p - nactive)))
        dual_feasible = np.dot(np.linalg.inv(B_sel_1.T), feasible)
        sel_prob = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                   noise_variance, tau,
                                                   epsilon)
        sel_prob_min = sel_prob.minimize_opt()
        return sel_prob_min[0] - np.true_divide(np.dot(mean.T, mean), 2 * noise_variance), \
               np.dot(B_sel_1.T, sel_prob_min[1])

    print test_minimizer_dual()

    def test_dual_compare_one_sparse():
        if nactive==1:
            snr_seq = np.linspace(-10, 10, num=20)
            lagrange = lam * np.ones(p)
            feasible = np.append(-np.fabs(np.random.standard_normal(nactive)),
                                 np.random.standard_normal((p - nactive)))
            dual_feasible = np.dot(np.linalg.inv(B_sel_1.T), feasible)
            print dual_feasible
            for i in range(snr_seq.shape[0]):
                parameter = snr_seq[i]
                mean = X_1[:, active].dot(parameter)
                print "parameter value", parameter
                sel = selection_probability(V, B_sel, gamma_sel, noise_variance, tau, lam, y, betaE, cube)
                sel_log_val = sel.optimization(parameter*np.ones(nactive),method="log_barrier")[0]-\
                              np.true_divide(np.dot(mean.T,mean),2*noise_variance)\
                              -np.true_divide(np.dot(gamma_sel[:nactive].T,gamma_sel[:nactive]),2*(tau**2))
                sel_prob = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                           noise_variance, tau, epsilon)
                sel_prob_min = sel_prob.minimize_opt()[0]-np.true_divide(np.dot(mean.T,mean),2*noise_variance)

                print "log selection probability", sel_log_val, sel_prob_min

    #test_dual_compare_one_sparse()


    def compare_objectives_dual():
        parameter = np.random.standard_normal(nactive)
        lagrange = lam * np.ones(p)
        mean = X_1[:, active].dot(parameter)
        feasible = np.append(-np.fabs(np.random.standard_normal(nactive)), np.random.standard_normal((p - nactive)))
        dual_feasible = np.dot(np.linalg.inv(B_sel_1.T), feasible)
        sel_prob = dual_selection_probability_func(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                   noise_variance, tau,
                                                   epsilon)
        break_up_1 = sel_prob.rand_CGF(dual_feasible), sel_prob.composed_barrier_conjugate(dual_feasible),\
               sel_prob.data_CGF(dual_feasible)
        sel_prob_2 = dual_selection_probability_only_objective(X_1, dual_feasible, active, active_signs, lagrange, mean,
                                                   noise_variance, randomization.isotropic_gaussian((p,), 1.),
                                                   epsilon)

        break_up_2 = sel_prob_2.smooth_objective(dual_feasible, mode='func', check_feasibility=False)

        print break_up_1
        print break_up_2


    #compare_objectives_dual()



