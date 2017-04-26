import os, numpy as np, pandas, statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso


if not os.path.exists("NRTI_DATA.txt"):
    NRTI = pandas.read_table("http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt", na_values="NA")
else:
    NRTI = pandas.read_table("NRTI_DATA.txt")

NRTI_specific = []
NRTI_muts = []
mixtures = np.zeros(NRTI.shape[0])
for i in range(1,241):
    d = NRTI['P%d' % i]
    for mut in np.unique(d):
        if mut not in ['-','.'] and len(mut) == 1:
            test = np.equal(d, mut)
            if test.sum() > 10:
                NRTI_specific.append(np.array(np.equal(d, mut)))
                NRTI_muts.append("P%d%s" % (i,mut))

NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)
print("here")

# Next, standardize the data, keeping only those where Y is not missing

X_NRTI = np.array(NRTI_specific, np.float)
Y = NRTI['3TC'] # shorthand
keep = ~np.isnan(Y).astype(np.bool)
X_NRTI = X_NRTI[np.nonzero(keep)]; Y=Y[keep]
Y = np.array(np.log(Y), np.float); Y -= Y.mean()
X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
X = X_NRTI # shorthand
n, p = X.shape
X /= np.sqrt(n)

ols_fit = sm.OLS(Y, X).fit()
sigma_3TC = np.linalg.norm(ols_fit.resid) / np.sqrt(n-p-1)
OLS_3TC = ols_fit.params

# Design matrix
# Columns are site / amino acid pairs


#solving the Lasso at theoretical lambda
tau = 1.0
print(tau**2)
random_Z = np.random.normal(loc=0.0, scale= tau, size= p)
sel = selection(X, Y, random_Z, sigma=sigma_3TC)

lam, epsilon, active, betaE, cube, initial_soln = sel

print("value of tuning parameter",lam)
print("nactive", active.sum())

active_set_0 = [NRTI_muts[i] for i in range(p) if active[i]]
print("active variables", active_set_0)
active_set = [i for i in range(p) if active[i]]

noise_variance = sigma_3TC**2
nactive = betaE.shape[0]
active_sign = np.sign(betaE)
feasible_point = np.fabs(betaE)
lagrange = lam * np.ones(p)

generative_X = X[:, active]
prior_variance = 1000.
randomizer = randomization.isotropic_gaussian((p,), 1.)

Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(Y))
post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
unadjusted_intervals = np.vstack([post_mean - 1.65*(post_var.diagonal()),post_mean + 1.65*(post_var.diagonal())])
unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])
#print(unadjusted_intervals)

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

#map = inf.map_solve(nstep = 500)[::-1]

toc = time.time()
samples = inf.posterior_samples()
tic = time.time()
print('sampling time', tic - toc)

adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
sel_mean = np.mean(samples, axis=0)
adjusted_intervals = np.vstack([sel_mean, adjusted_intervals])

print("active variables", active_set_0)
print("selective mean", sel_mean)
#print("selective map", map[1])
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
axes.set_ylim([-8, 70])             # y-axis bounds

ax.set_ylabel(' ')
ax.set_title('selected variables'.format(active_set))
ax.set_xticks(ind + 1.2* width)

ax.set_xticklabels(active_set_0, rotation=90)


#ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6'))

ax.legend((rects1[0], rects2[0]), ('Unadjusted', 'Adjusted'), loc='upper left')

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

plt.savefig('/Users/snigdhapanigrahi/Results_bayesian/credible_hiv_selected_0.pdf', bbox_inches='tight')

##################################################
ind = np.zeros(len(active_set), np.bool)

index = active_set_0.index('P184V')
ind[index] = 1

active_set_0.pop(index)

active_set.pop(index)

intervals = intervals[:, ~ind]


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
axes.set_ylim([-8, 12])             # y-axis bounds

ax.set_ylabel(' ')
ax.set_title('selected variables'.format(active_set))
ax.set_xticks(ind + 1.2* width)

ax.set_xticklabels(active_set_0, rotation=90)

ax.legend((rects1[0], rects2[0]), ('Unadjusted', 'Adjusted'), loc='upper right')

print('here')

plt.savefig('/Users/snigdhapanigrahi/Results_bayesian/credible_hiv_selected_1.pdf', bbox_inches='tight')