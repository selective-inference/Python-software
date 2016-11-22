from __future__ import print_function
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.tests.decorators import wait_for_return_value
from selection.bayesian.initial_soln import selection
from selection.bayesian.selection_probability_rr import cube_subproblem, cube_gradient, cube_barrier, \
    selection_probability_objective, cube_subproblem_scaled, cube_gradient_scaled, cube_barrier_scaled, \
    cube_subproblem_scaled
from selection.randomized.api import randomization
from selection.bayesian.selection_probability import selection_probability_methods
from selection.bayesian.dual_scipy import dual_selection_probability_func
from selection.bayesian.inference_rr import sel_prob_gradient_map, selective_map_credible
from selection.bayesian.inference_fs import sel_prob_gradient_map_fs, selective_map_credible_fs
from selection.bayesian.inference_ms import sel_prob_gradient_map_ms, selective_map_credible_ms

def test_inf_regreg():
    n = 100
    p = 50
    s = 5
    snr = 3

    # sampling the Gaussian instance
    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z = np.random.standard_normal(p)
    # getting randomized Lasso solution
    sel = selection(X_1, y, random_Z)

    lam, epsilon, active, betaE, cube, initial_soln = sel
    print(epsilon, lam, betaE, true_beta, active, active.sum())
    noise_variance = 1.
    nactive = betaE.shape[0]
    active_signs = np.sign(betaE)
    tau = 1  # randomization_variance

    primal_feasible = np.fabs(betaE)
    dual_feasible = np.ones(p)
    dual_feasible[:nactive] = -np.fabs(np.random.standard_normal(nactive))
    lagrange = lam * np.ones(p)
    generative_X = X_1[:, active]
    prior_variance = 1000.

    Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
    post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
    post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
    unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])
    unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])
    #print(np.vstack([post_mean, unadjusted_intervals]))

    inf_rr = selective_map_credible(y,
                                    X_1,
                                    primal_feasible,
                                    dual_feasible,
                                    active,
                                    active_signs,
                                    lagrange,
                                    generative_X,
                                    noise_variance,
                                    prior_variance,
                                    randomization.isotropic_gaussian((p,), tau),
                                    epsilon)

    map = inf_rr.map_solve_2(nstep = 100)[::-1]
    print("selective map", map[1])

    #print ("gradient at map", -inf_rr.smooth_objective(map[1], mode='grad'))
    #print ("map objective, map", map[0], map[1])
    toc = time.time()
    samples = inf_rr.posterior_samples()
    tic = time.time()
    print('sampling time', tic - toc)
    adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
    adjusted_intervals = np.vstack([map[1], adjusted_intervals])
    print(active)
    print("selective map and intervals", adjusted_intervals)
    print("usual posterior based map & intervals", unadjusted_intervals)
    return nactive, np.vstack([unadjusted_intervals, adjusted_intervals])

test = test_inf_regreg()
intervals = test[1]
nactive = test[0]

un_mean = intervals[0,:]
un_lower_error = list(un_mean-intervals[1,:])
un_upper_error = list(intervals[2,:]-un_mean)
unStd = [un_lower_error, un_upper_error]
ad_mean = intervals[3,:]
ad_lower_error = list(ad_mean-intervals[4,:])
ad_upper_error = list(intervals[5,:]- ad_mean)
adStd = [ad_lower_error, ad_upper_error]


N = len(un_mean)               # number of data entries
ind = np.arange(N)              # the x locations for the groups
width = 0.35                    # bar width

print('here')

fig, ax = plt.subplots()

rects1 = ax.bar(ind, un_mean,                  # data
                width,                          # bar width
                color='darkgrey',        # bar colour
                yerr=unStd,  # data for error bars
                error_kw={'ecolor':'dimgrey',    # error-bars colour
                          'linewidth':2})       # error-bar width

rects2 = ax.bar(ind + width, ad_mean,
                width,
                color='thistle',
                yerr=adStd,
                error_kw={'ecolor':'darkmagenta',
                          'linewidth':2})

axes = plt.gca()
axes.set_ylim([-6, 10])             # y-axis bounds

ax.set_ylabel('Credible')
ax.set_title('Credible and Map')
ax.set_xticks(ind + width)
if nactive == 1:
    ax.set_xticklabels(('Coef1'))
elif nactive == 2:
    ax.set_xticklabels(('Coef1', 'Coef2'))
elif nactive == 3:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3'))
elif nactive == 4:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4'))
elif nactive == 5:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5'))
elif nactive == 6:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6'))
elif nactive == 7:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6', 'Coef7'))
elif nactive == 8:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6', 'Coef7','Coef8'))
elif nactive == 9:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6', 'Coef7','Coef8','Coef9'))
else:
    ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6', 'Coef7', 'Coef8', 'Coef9','Coef10'))


#ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6'))

ax.legend((rects1[0], rects2[0]), ('Unadjusted', 'Adjusted'))

print('here')

#def autolabel(rects):
#    for rect in rects:
#        height = rect.get_height()
#        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                '%d' % int(height),
#                ha='center',            # vertical alignment
#                va='bottom'             # horizontal alignment
#                )

#autolabel(rects1)
#autolabel(rects2)

#plt.show()                              # render the plot

plt.savefig('/Users/snigdhapanigrahi/Documents/Research/Python_plots/credible_un_adjusted_1.pdf', bbox_inches='tight')

