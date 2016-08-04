import numpy as np
from scipy.stats import t as tdist, norm as ndist
import scipy.stats
import scipy.optimize
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm

n = 500
kappa = 0.8
#sd = 1
truth = 0

def logistic(n=1):
    U = np.random.sample(n)
    return np.log(U / (1 - U))

def Gbar(z):
    return np.exp(-z) / (1 + np.exp(-z))

def selection_event(S, threshold):
    # ttest_1samp returns t-statistic([0]) and two sided p-value([1])
    return scipy.stats.ttest_1samp(S, 0)[0] + kappa*logistic() > threshold

# returns a sequence of bootstraped means
def bootstrap(sample, nsample=5000):
    boot = []
    n = sample.shape[0]
    for _ in range(nsample):
        bootS = np.random.choice(sample, n, replace=True)
        boot.append(bootS.mean())
    return np.array(boot)


# plugs in pivot function here for dbn_func. then t=(observed-theta)/sd and delta=sqrt(n)*theta/sd
#for sd here we plug in S.std()
def interval(dbn_func, observed, sd, n, Z, coverage=0.9):
    f1 = lambda theta: dbn_func((observed - theta) / sd, np.sqrt(n) * theta/sd , Z) - (1 - coverage) / 2
    # finds a zero of the function within an interval
    v1 = scipy.optimize.bisect(f1, observed - 6 * sd, observed + 6 * sd )
    f2 = lambda theta: dbn_func((observed - theta) / sd, np.sqrt(n) * theta/sd, Z) - (1 - (1 - coverage) / 2)
    v2 = scipy.optimize.bisect(f2, observed - 6 * sd, observed + 6 * sd )
    return v2, v1


def noise_normal(n):
    # this noise has exponential moments...
    Z = np.random.standard_normal(n)
    return Z

#noise_sd = np.std(noise(50000))

def simulation(n, noise='normal', threshold=2, nsample=5000, coverage=0.90):
    count = 0
    # loops until it finds a sample S for which the selection event is true
    while True:
        count += 1
        if (noise=='normal'):
            S = noise_normal(n)+truth
        if (noise=='laplace'): #Laplace(loc, scale) has variance 2*scale^2
            S = np.random.laplace(loc=0, scale= 1./np.sqrt(2),size=n)+truth
        if (noise=='uniform'): #Unif(a,b) has variance (b-a)^2/12
            S = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=n)+truth
        if (noise=='logistic'): # Logistic(loc, scale) has variance scale^2*pi^2/3=1
            S = np.random.logistic(loc=0, scale=np.sqrt(3)/np.pi, size=n)
        if selection_event(S, threshold):
            break

    boot_sample = bootstrap(S, nsample)
    bootZ = boot_sample - S.mean()
    bootZ = bootZ/boot_sample.std() # forms a vector of \sqrt{n}(\bar{Y^*}-\bar{Y}) of length of B 
    
    gaussZ = np.random.standard_normal(2000)
    
    def pivot(t, delta, Z):
        num = (Gbar((threshold - Z - delta) / kappa) * (Z <= t)).mean()
        den = (Gbar((threshold - Z - delta) / kappa)).mean()
        return num / den
    
    bootL, bootU = interval(pivot, S.mean(), S.std()/np.sqrt(n), n, bootZ, coverage=coverage)
    
    gaussL, gaussU = interval(pivot, S.mean(), 1/np.sqrt(n), n, gaussZ, coverage=coverage)
    boot_cover = (bootL < truth) * (bootU > truth)
    gauss_cover = (gaussL < truth) * (gaussU > truth)
    return pivot((S.mean() - truth) / boot_sample.std(), np.sqrt(n) * truth/S.std(), bootZ), count, pivot, boot_cover, bootU-bootL, (bootU - bootL) / (2 * np.fabs(ndist.ppf((1 - coverage) / 2.)) * np.std(S) / np.sqrt(n))

    
#p, count, pivot, covered, length, ratio = simulation(n)
#print p, covered, length, count


random.seed(1)
fig=plt.figure()
fig.suptitle('P values for the simple example')


for noise in ['normal','laplace', 'uniform','logistic']:
    P = []
    C = []
    coverage = []
    num_except = 0
    length = []
    ratio = []
    for i in range(500):
        #print i
        try:
            result = simulation(n, noise=noise)
        except ValueError: # probably root finding failed
            num_except += 1
        P.append(result[0])
        C.append(result[1])
        coverage.append(result[3])
        length.append(result[4])
        ratio.append(result[5])
    print noise    
    print 'probability of selection:', 1. / np.mean(C)
    print 'coverage:', np.mean(coverage)
    print 'number of root finding fails:', num_except
    print 'average ratio of length to T interval:', np.mean(ratio)
    print 'average length of interval:', np.mean(length)
    # generates one plot for the p-values for all types of errors
    if (noise=='normal'):
        plot1=fig.add_subplot(221)
        plot1.set_title('Normal errors')
    if (noise=='laplace'):
        plot1=fig.add_subplot(222)
        plot1.set_title('Laplace errors')
    if (noise=='uniform'):
        plot1=fig.add_subplot(223)
        plot1.set_title('Uniform errors')
    if (noise=='logistic'):
        plot1=fig.add_subplot(224)
        plot1.set_title('Logistic errors')
    ecdf = sm.distributions.ECDF(P)
    x = np.linspace(min(P), max(P))
    y = ecdf(x)
    plt.plot(x, y, lw=2)
    plt.plot([0,1], [0,1], 'k-', lw=1)
plt.savefig('foo.pdf')    
plt.show()




