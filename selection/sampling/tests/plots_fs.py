from __future__ import print_function
import numpy as np
from .test_fstep_langevin import test_fstep
from .test_kfstep import test_kfstep
from matplotlib import pyplot as plt
from scipy.stats import probplot, uniform
import random
import statsmodels.api as sm

def main():

    random.seed(4)

    fig = plt.figure()
    plot_1step = fig.add_subplot(121)
    plot_kstep = fig.add_subplot(122)


    P0 = []
    for i in range(300):

        print("iteration", i)
        p0 = test_fstep(Langevin_steps=10000, burning=2000)
        P0.append(p0)

    print("one step FS done! mean: ", np.mean(P0), "std: ", np.std(P0))
    #probplot(P0, dist=uniform, sparams=(0,1), plot=plot_1step, fit=False)
    #plot_1step.plot([0, 1], color='k', linestyle='-', linewidth=2)

    ecdf = sm.distributions.ECDF(P0)
    x = np.linspace(min(P0), max(P0))
    y = ecdf(x)
    plot_1step.plot(x, y, '-o',lw=2)
    plot_1step.plot([0, 1], [0, 1], 'k-', lw=2)

    plot_1step.set_title("One step FS")
    plot_1step.set_xlim([0,1])
    plot_1step.set_ylim([0,1])


    P0 = []
    for i in range(300):
        print("iteration", i)
        p0 = test_kfstep(Langevin_steps=10000, burning=2000)
        P0.append(p0)

    print("k steps FS done done! mean: ", np.mean(P0), "std: ", np.std(P0))
    #probplot(P0, dist=uniform, sparams=(0,1), plot=plot_kstep, fit=False)
    #plot_kstep.plot([0, 1], color='k', linestyle='-', linewidth=2)


    ecdf = sm.distributions.ECDF(P0)
    x = np.linspace(min(P0), max(P0))
    y = ecdf(x)
    plot_kstep.plot(x, y,'-o', lw=2)
    plot_kstep.plot([0, 1], [0, 1], 'k-', lw=2)

    plot_kstep.set_title("Four steps FS")
    plot_kstep.set_xlim([0,1])
    plot_kstep.set_ylim([0,1])



    plt.show()
    plt.savefig('FS_Langevin.pdf')


