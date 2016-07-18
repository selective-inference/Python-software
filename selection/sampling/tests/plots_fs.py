import numpy as np
from test_fstep_langevin import test_fstep
from test_kfstep import test_kfstep
from matplotlib import pyplot as plt
from scipy.stats import probplot, uniform
import random

random.seed(1)

fig = plt.figure()
plot_1step = fig.add_subplot(121)
plot_kstep = fig.add_subplot(122)


P0 = []
for i in range(200):
    print "iteration", i
    p0 = test_fstep()
    P0.append(p0)

print "one step FS done! mean: ", np.mean(P0), "std: ", np.std(P0)
probplot(P0, dist=uniform, sparams=(0,1), plot=plot_1step, fit=False)
plot_1step.plot([0, 1], color='k', linestyle='-', linewidth=2)
plot_1step.set_title("One step FS")
plot_1step.set_xlim([0,1])
plot_1step.set_ylim([0,1])


P0 = []
for i in range(200):
    print "iteration", i
    p0 = test_kfstep(k=3)
    P0.append(p0)

print "k steps FS done done! mean: ", np.mean(P0), "std: ", np.std(P0)
probplot(P0, dist=uniform, sparams=(0,1), plot=plot_kstep, fit=False)
plot_kstep.plot([0, 1], color='k', linestyle='-', linewidth=2)
plot_kstep.set_title("Three steps FS")
plot_kstep.set_xlim([0,1])
plot_kstep.set_ylim([0,1])



plt.show()

