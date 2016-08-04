import numpy as np
from test_lasso_fixedX_langevin import test_lasso as test_fixedX
from test_logistic_langevin import test_lasso as test_logistic
from test_lasso_randomX_langevin import test_lasso as test_randomX
from matplotlib import pyplot as plt
from scipy.stats import probplot, uniform

np.random.seed(2)

fig = plt.figure()
plot_randomX = fig.add_subplot(131)
plot_fixedX = fig.add_subplot(132)
plot_logistic = fig.add_subplot(133)

P0, PA = [], []

for i in range(100):
    print "iteration", i
    p0, pA = test_randomX(s=5, n=200, p=20, Langevin_steps=7000)
    P0.extend(p0); PA.extend(pA)

print "random X done! mean: ", np.mean(P0), "std: ", np.std(P0)
probplot(P0, dist=uniform, sparams=(0,1), plot=plot_randomX, fit=False)
plot_randomX.plot([0, 1], color='k', linestyle='-', linewidth=2)
plot_randomX.set_title("Lasso random X")
plot_randomX.set_xlim([0,1])
plot_randomX.set_ylim([0,1])


P0, PA = [], []
for i in range(100):
    print "iteration", i
    p0, pA = test_fixedX(s=5, n=200, p=20, Langevin_steps=7000)
    P0.extend(p0); PA.extend(pA)

print "fixed X done! mean: ", np.mean(P0), "std: ", np.std(P0)
probplot(P0, dist=uniform, sparams=(0,1), plot=plot_fixedX, fit=False)
plot_fixedX.plot([0, 1], color='k', linestyle='-', linewidth=2)
plot_fixedX.set_title("Lasso fixed X")
plot_fixedX.set_xlim([0,1])
plot_fixedX.set_ylim([0,1])


P0, PA = [], []
for i in range(100):
    print "iteration", i
    p0, pA = test_logistic(s=5, n=200, p=20, Langevin_steps=7000)
    P0.extend(p0); PA.extend(pA)
print "logistic done! mean: ", np.mean(P0), "std: ", np.std(P0)
probplot(P0, dist=uniform, sparams=(0,1), plot=plot_logistic, fit=False)
plot_logistic.plot([0, 1], color='k', linestyle='-', linewidth=2)
plot_logistic.set_title("Logistic random X")
plot_logistic.set_xlim([0,1])
plot_logistic.set_ylim([0,1])


plt.show()

