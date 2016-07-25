import numpy as np
from bayes_boot_randomX import test_lasso
from matplotlib import pyplot as plt
from scipy.stats import laplace, probplot, uniform


for steps in [1, 2, 4, 6]:
    
    plt.figure()
    print steps*(10**4)
    #plt.ion()
    P0, PA = [], []
    for i in range(100):
        print "iteration", i
        p0, pA = test_lasso(Langevin_steps=steps*(10**4))
        if np.sum(p0)>-1:
            P0.extend(p0); PA.extend(pA)
        #plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)


    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)

    #while True:
    #    plt.pause(0.05)
    plt.savefig('bayes'+str(steps)+'.pdf')




