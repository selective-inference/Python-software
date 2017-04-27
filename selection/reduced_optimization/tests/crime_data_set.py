import os, numpy as np, pandas, statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso


NRTI =  pandas.read_table("/Users/snigdhapanigrahi/Results_bayesian/trimmed_communities.txt", sep='\s+')
print("shape of data", NRTI.shape)

NRTI = NRTI.as_matrix()

X_NRTI = np.delete(NRTI, 25, 1)
print("shape",X_NRTI.shape)

X_NRTI = X_NRTI.astype(float)
Y = NRTI[:,97]

Y = Y.astype(float)
X_NRTI -= X_NRTI.mean(0)[None, :]
X = X_NRTI
n, p = X.shape
X /= (X.std(0)[None, :] * np.sqrt(n))
#print(X.T.dot(X))
#X /= np.sqrt(n)

ols_fit = sm.OLS(Y, X).fit()
print("residual", np.linalg.norm(ols_fit.resid))
sigma_3TC = np.linalg.norm(ols_fit.resid) / np.sqrt(n-p-1)
OLS_3TC = ols_fit.params
print("sigma", sigma_3TC)

tau = 0.5
print(tau**2)
random_Z = np.random.normal(loc=0.0, scale= tau, size= p)
sel = selection(X, Y, random_Z, sigma=sigma_3TC)

lam, epsilon, active, betaE, cube, initial_soln = sel

print("value of tuning parameter",lam)
print("nactive", active.sum())

active_set_0 = [i for i in range(p) if active[i]]
print("active variables", active_set_0)
active_set = [i for i in range(p) if active[i]]

noise_variance = sigma_3TC**2
nactive = betaE.shape[0]
active_sign = np.sign(betaE)
feasible_point = np.fabs(betaE)
lagrange = lam * np.ones(p)

generative_X = X[:, active]
prior_variance = 1000.
randomizer = randomization.isotropic_gaussian((p,), 0.5)

Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(Y))
post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])
unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])

grad_map = sel_prob_gradient_map_lasso(X,
                                       feasible_point,
                                       active,
                                       active_sign,
                                       lagrange,
                                       generative_X,
                                       noise_variance,
                                       randomizer,
                                       epsilon)

inf = selective_inf_lasso(Y, grad_map, prior_variance)

toc = time.time()
samples = inf.posterior_samples()
tic = time.time()
print('sampling time', tic - toc)

adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
sel_mean = np.mean(samples, axis=0)
adjusted_intervals = np.vstack([sel_mean, adjusted_intervals])

print("active variables", active_set_0)
print("selective mean", sel_mean)
print("selective map and intervals", adjusted_intervals)
print("usual posterior based map & intervals", unadjusted_intervals)

intervals = np.vstack([unadjusted_intervals, adjusted_intervals])

###################################################################################

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

width_0 = 0.10

print('here')

fig, ax = plt.subplots()

rects1 = ax.bar(ind, un_mean,                  # data
                width,                          # bar width
                color='royalblue',        # bar colour
                yerr=unStd,  # data for error bars
                error_kw={'ecolor':'darkblue',    # error-bars colour
                          'linewidth':2})       # error-bar width

rects2 = ax.bar(ind + width, ad_mean,
                width,
                color='red',
                yerr=adStd,
                error_kw={'ecolor':'maroon',
                          'linewidth':2})

axes = plt.gca()
axes.set_ylim([-20, 20])             # y-axis bounds

ax.set_ylabel(' ')
ax.set_title('selected variables'.format(active_set))
ax.set_xticks(ind + 1.2* width)

ax.set_xticklabels(active_set_0, rotation=90)

ax.legend((rects1[0], rects2[0]), ('Unadjusted', 'Adjusted'), loc='upper left')

print('here')

plt.savefig('/Users/snigdhapanigrahi/Results_bayesian/credible_crime.pdf', bbox_inches='tight')
