import numpy as np
import matplotlib.pyplot as plt

un_mean = 5*np.ones(3)
un_lower_error = un_mean-[3.63, 2.22, 5.55]
un_upper_error = [7.30, 5.61, 9.11]-un_mean
unStd = [un_lower_error, un_upper_error]
ad_mean = 5*np.ones(3)
ad_lower_error = ad_mean-[3.60, 1.99, 5.63]
ad_upper_error = [7.12, 5.54, 9.16]- ad_mean
adStd = [ad_lower_error, ad_upper_error]


N = len(un_mean)               # number of data entries
ind = np.arange(N)              # the x locations for the groups
width = 0.25                    # bar width

print('here')

fig, ax = plt.subplots()

rects1 = ax.bar(ind, un_mean,                  # data
                width,                          # bar width
                color='MediumSlateBlue',        # bar colour
                yerr=unStd,  # data for error bars
                error_kw={'ecolor':'Tomato',    # error-bars colour
                          'linewidth':2})       # error-bar width

rects2 = ax.bar(ind + width, ad_mean,
                width,
                color='Tomato',
                yerr=adStd,
                error_kw={'ecolor':'MediumSlateBlue',
                          'linewidth':2})

axes = plt.gca()
axes.set_ylim([0, 10])             # y-axis bounds

ax.set_ylabel('Credible')
ax.set_title('Credible and map')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Coef1', 'Coef2', 'Coef3'))

ax.legend((rects1[0], rects2[0]), ('Unadjusted', 'Adjusted'))

print('here')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center',            # vertical alignment
                va='bottom'             # horizontal alignment
                )

autolabel(rects1)
autolabel(rects2)

plt.show()                              # render the plot
