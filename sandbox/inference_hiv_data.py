from __future__ import print_function
import os, numpy as np, pandas, statsmodels.api as sm
import time
import regreg.api as rr
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_via_approx_density import approximate_conditional_density
from selection.approx_ci.estimator_approx import M_estimator_approx

from selection.randomized.query import naive_confidence_intervals
from selection.api import randomization
import matplotlib.pyplot as plt


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

lam_frac = 1.
loss = rr.glm.gaussian(X, Y)
epsilon = 1. / np.sqrt(n)
lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_3TC
print(lam)

W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p),weights=dict(zip(np.arange(p), W)), lagrange=1.)

randomization = randomization.isotropic_gaussian((p,), scale=1.)

M_est = M_estimator_approx(loss, epsilon, penalty, randomization, randomizer='gaussian')
M_est.solve_approx()
active = M_est._overall
active_set = np.asarray([i for i in range(p) if active[i]])
nactive = np.sum(active)

active_set_0 = [NRTI_muts[i] for i in range(p) if active[i]]

ci_active = np.zeros((nactive, 2))
ci_length = np.zeros(nactive)
mle_active = np.zeros((nactive,1))

ci = approximate_conditional_density(M_est)
ci.solve_approx()

class target_class(object):
    def __init__(self, target_cov):
        self.target_cov = target_cov
        self.shape = target_cov.shape


target = target_class(M_est.target_cov)
ci_naive = naive_confidence_intervals(target, M_est.target_observed)

for j in range(nactive):
    ci_active[j, :] = np.array(ci.approximate_ci(j))
    ci_length[j] = ci_active[j,1] - ci_active[j,0]
    mle_active[j, :] = ci.approx_MLE_solver(j, nstep=100)[0]

unadjusted_mle = np.zeros((nactive,1))
for j in range(nactive):
    unadjusted_mle[j, :] = ci.target_observed[j]

adjusted_intervals = np.hstack([mle_active, ci_active]).T
unadjusted_intervals = np.hstack([unadjusted_mle, ci_naive]).T

print("adjusted confidence", adjusted_intervals)
print("naive confidence", unadjusted_intervals)

intervals = np.vstack([unadjusted_intervals, adjusted_intervals])

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
axes.set_ylim([-6, 60])             # y-axis bounds

ax.set_ylabel('Credible')
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

plt.savefig('/Users/snigdhapanigrahi/Documents/Research/Python_plots/icml_hiv_plots.pdf', bbox_inches='tight')

##################################################
ind = np.zeros(len(active_set), np.bool)

index = active_set_0.index('P184V')
ind[index] = 1

active_set_0.pop(index)

active_set = [i for i in range(p) if active[i]]
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
axes.set_ylim([-6, 12])             # y-axis bounds

ax.set_ylabel('Credible')
ax.set_title('selected variables'.format(active_set))
ax.set_xticks(ind + 1.2* width)

ax.set_xticklabels(active_set_0, rotation=90)

ax.legend((rects1[0], rects2[0]), ('Unadjusted', 'Adjusted'), loc='upper right')

print('here')

plt.savefig('/Users/snigdhapanigrahi/Documents/Research/Python_plots/icml_hiv_plots_0.pdf', bbox_inches='tight')