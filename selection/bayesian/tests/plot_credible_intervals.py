import numpy as np
import matplotlib.pyplot as plt

un_mean = [3.49, 3.16, 0.78, 1.33, -2.12, -2.23]
un_mean = [float(i) for i in un_mean]
un_lower_error = list(np.array(un_mean)-np.array([1.74, 1.46, -0.98, -0.39, -3.81, -3.92]))
un_upper_error = list(np.array([5.25, 4.86, 2.54, 3.05, -0.43, -0.53])-np.array(un_mean))
unStd = [un_lower_error, un_upper_error]
ad_mean = [3.09, 3.04, -0.63, 0.26, -1.32, -1.45]
ad_mean = [float(i) for i in ad_mean]
ad_lower_error = list(np.array(ad_mean)-np.array([1.39, 1.12, -3.15, -2.01, -3.44, -3.22]))
ad_upper_error = list(np.array([5.26, 5.18, 1.66, 2.25, 1.10, 0.96])- np.array(ad_mean))
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
axes.set_ylim([-5, 7])             # y-axis bounds

ax.set_ylabel('Credible')
ax.set_title('Credible and Map')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3', 'Coef4', 'Coef5', 'Coef6'))

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

plt.savefig('/Users/snigdhapanigrahi/Documents/Research/Python_plots/credible_un_adjusted.pdf', bbox_inches='tight')
