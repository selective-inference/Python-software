import numpy as np

from scipy.stats import laplace, probplot, uniform

from selection.algorithms.lasso import instance
import selection.sampling.randomized.api as randomized
from matplotlib import pyplot as plt
from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin
from scipy.integrate import quad


def test_fstep(s=0, n=1000, p=5):

    X, y, _, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, sigma=1.,rho=0)

    random_Z = np.random.logistic(loc=0, scale=1., size=p)
    T = np.dot(X.T,y)
    T_random = T + random_Z
    T_abs = np.abs(T_random)
    idx = np.argmax(T_abs)
    sign = np.sign(T_random[idx])



    def bootstrap_samples(X=X, y=y, idx=idx):
        nsample = 200
        boot_samples = []
        n = X.shape[0]
        for _ in range(nsample):
            indices = np.random.choice(n, size=(n,), replace=True)
            y_star = y[indices]
            boot_samples.append(np.inner(X[:, idx], y_star))
        return boot_samples

    boot_samples = bootstrap_samples()

    def integrand(w, boot_sample, T = T,
                  sign = sign, idx = idx):

        value = sign*(boot_sample-T[idx]+w)
        if value<0:
            print 'value'
        p = T.shape[0]
        product = 1
        for i in range(p):
            if (i!=idx):
                rhs = value - T[i]
                lhs = - value - T[i]
                product *= (1./(1+np.exp(-rhs))-1./(1+np.exp(-lhs)))
        product *= np.exp(-w)/(np.square(1+np.exp(-w)))
        # print 'product', product
        return product

    #print 'integrand', integrand(0, boot_samples[0])

    num = 0
    den = 0
    for i in range(len(boot_samples)):
        lower = - 20
        upper = 20
        if (sign>0):
            lower = - (boot_samples[i]-T[idx])
            #print lower, upper
        else:
            upper = - (boot_samples[i]-T[idx])
            #print lower, upper
        sel_prob = quad(integrand, lower, upper, args=(boot_samples[i]))[0]
        den +=sel_prob
        if (T[idx] <= boot_samples[i]-T[idx]):
            num += sel_prob

    pval = np.true_divide(num, den)
    pval = 2*min(pval, 1-pval)
    print pval
    return pval


if __name__ == "__main__":

    P0 =[]
    for i in range(20):
        print "iteration", i
        p0 = test_fstep()
        P0.append(p0)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.show()