import numpy as np
from selection.sampling.randomized.tests.bootstrap.bayes_boot_randomX import test_lasso
from matplotlib import pyplot as plt
from scipy.stats import probplot, uniform

np.random.seed(1)

fig = plt.figure()
plot_normal = fig.add_subplot(221)
plot_uniform = fig.add_subplot(222)
plot_laplace = fig.add_subplot(223)
plot_logistic = fig.add_subplot(224)
import statsmodels.api as sm


for noise in ["normal", "uniform", "laplace", "logistic"]:
    P0, PA = [], []

    for i in range(1000):
        print "iteration", i, noise
        p0, pA = test_lasso(randomization_dist="logistic")
        if np.sum(p0)>-1:
            P0.extend(p0); PA.extend(pA)
    print "bootstrap for "+noise+" done! mean: ", np.mean(P0), "std: ", np.std(P0)
    ecdf = sm.distributions.ECDF(P0)
    x = np.linspace(min(P0), max(P0))
    y = ecdf(x)
    if noise=="normal":
        plot_normal.plot(x, y, lw=2)
        plot_normal.plot([0, 1], [0, 1], 'k-', lw=1)
        plot_normal.set_title(noise)
        plot_normal.set_xlim([0, 1])
        plot_normal.set_ylim([0, 1])
    if noise =="uniform":
        plot_uniform.plot(x, y, lw=2)
        plot_uniform.plot([0, 1], [0, 1], 'k-', lw=1)
        plot_uniform.set_title(noise)
        plot_uniform.set_xlim([0,1])
        plot_uniform.set_ylim([0, 1])
    if noise == "laplace":
        plot_laplace.plot(x, y, lw=2)
        plot_laplace.plot([0, 1], [0, 1], 'k-', lw=1)
        plot_laplace.set_title(noise)
        plot_laplace.set_xlim([0, 1])
        plot_laplace.set_ylim([0, 1])
    if noise == "logistic":
        plot_logistic.plot(x, y, lw=2)
        plot_logistic.plot([0, 1], [0, 1], 'k-', lw=1)
        plot_logistic.set_title(noise)
        plot_logistic.set_xlim([0, 1])
        plot_logistic.set_ylim([0, 1])


plt.savefig('wild_bootstrap_plot_logistic.pdf')

