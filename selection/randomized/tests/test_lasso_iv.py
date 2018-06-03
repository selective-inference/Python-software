import numpy as np

import selection.randomized.lasso as L; reload(L)
<<<<<<< HEAD
from selection.randomized.lasso_iv import lasso_iv, lasso_iv_ar, group_lasso_iv
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

################################################################################

#  test file for lasso_iv class

# include the screening in here

################################################################################

# if true_model is True, Sigma_12 is the true Sigma_{12}
# otherwise Sigma_12 will be the consistent estimator
def test_group_lasso_iv_instance(n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1., marginalize=False):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv(Y,D,Z)
    passed = conv.fit()
    if not passed:
        return None, None

    if true_model is True:
        sigma_11 = 1.
        sigma_12 = Sigma_12
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]
        sigma_12 = Sigma_matrix[0,1]

    pivot = None
    interval = None

    pivot, _, interval = conv.summary(parameter=beta_star, Sigma_11=sigma_11, Sigma_12=sigma_12)

    return pivot, interval


def test_pivots_group_lasso(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1., marginalize=False):
    P0 = []
    #intervals = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval= test_group_lasso_iv_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star, marginalize=marginalize)
        if p0 is not None:
            P0.extend(p0)
            #intervals.extend(interval)
            coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            lengths.extend([interval[0][1] - interval[0][0]])

    print(len(P0), ' instances passing pre-test out of ', nsim, ' total instances')
    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths))

    return P0


def naive_pre_test_instance(n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv(Y,D,Z)

    if true_model is True:
        sigma_11 = 1.
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]

    pval, interval = conv.naive_inference(parameter=beta_star, Sigma_11 = sigma_11, compute_intervals=True)
    
    return pval, interval


def test_pivots_naive_pre_test(nsim=500, n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):
    P0 = []
    #intervals = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval= naive_pre_test_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        if p0 is not None:
            P0.extend([p0])
            #intervals.extend(interval)
            coverages.extend([(interval[0] < beta_star) * (interval[1] > beta_star)])
            lengths.extend([interval[1] - interval[0]])

    print(len(P0), ' instances passing pre-test out of ', nsim, ' total instances')
    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths))

    return P0


def plain_tsls_instance(n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv(Y,D,Z)

    if true_model is True:
        sigma_11 = 1.
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]

    pval, interval = conv.plain_inference(parameter=beta_star, Sigma_11 = sigma_11, compute_intervals=True)
    
    return pval, interval


def test_pivots_plain_tsls(nsim=500, n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):
    P0 = []
    #intervals = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval= plain_tsls_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        if p0 is not None:
            P0.extend([p0])
            #intervals.extend(interval)
            coverages.extend([(interval[0] < beta_star) * (interval[1] > beta_star)])
            lengths.extend([interval[1] - interval[0]])

    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths))

    return P0



# if true_model is True, Sigma_12 is the true Sigma_{12}
# otherwise Sigma_12 will be the consistent estimator
def test_lasso_iv_instance(n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., marginalize=False):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv(Y, D, Z)
    if marginalize:
        conv.fit_for_marginalize()
    else:
        conv.fit()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    interval = None
    power = None
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0]) and conv._inactive.sum()>0:
        pivot, _, interval, power = conv.summary(parameter=beta_star, Sigma_11=sigma_11, compute_power=True)
    return pivot, interval, power

def test_pivots(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., marginalize=False):
    P0 = []
    #intervals = []
    coverages = []
    lengths = []
    powers = []
    for i in range(nsim):
        p0, interval, power = test_lasso_iv_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star, marginalize=marginalize)
        if p0 is not None and interval is not None:
            P0.extend(p0)
            #intervals.extend(interval)
            coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            lengths.extend([interval[0][1] - interval[0][0]])
            powers.append(power)

    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths), np.std(lengths))
    print('powers', np.mean(np.array(powers), axis=0))

    #U = np.linspace(0, 1, 101)
    #plt.plot(U, ECDF(P0)(U))
    #plt.plot(U, U, 'k--')
    #plt.show()

    #return P0
    return lengths


# if true_model is True, Sigma_12 is the true Sigma_{12}
# otherwise Sigma_12 will be the consistent estimator
def test_lasso_iv_ar_instance(n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1.):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv_ar.bigaussian_instance(n=n,p=p,s=s, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv_ar(Y, D, Z)
    conv.fit()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    coverage = None
    power = None
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0]) and conv._inactive.sum()>0:
        pivot, _, coverage, power = conv.summary(parameter=beta_star, Sigma_11=sigma_11, compute_power=True)
    return pivot, coverage, power

def test_pivots_ar(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1.):
    P0 = []
    #intervals = []
    coverages = []
    #lengths = []
    powers = []
    for i in range(nsim):
        p0, coverage, power = test_lasso_iv_ar_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star)
        #if p0 is not None and interval is not None:
        if p0 is not None:
            P0.extend(p0)
            coverages.extend([coverage])
            powers.append(power)
            #intervals.extend(interval)
            #coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            #lengths.extend([interval[0][1] - interval[0][0]])

    print('pivots', np.mean(P0), np.std(P0), 1.-np.mean(np.array(P0) < 0.05))
    print('coverage', np.mean(coverages))
    print('powers', np.mean(np.array(powers), axis=0))
    #print('confidence intervals', np.mean(coverages), np.mean(lengths))

    #U = np.linspace(0, 1, 101)
    #plt.plot(U, ECDF(P0)(U))
    #plt.plot(U, U, 'k--')
    #plt.show()

    return P0



# Sigma_12 is the true Sigma_{12}
def test_stat_lasso_iv_instance(n=1000, p=10, s=3, ndraw=10000, burnin=2000, Sigma_12=0.8, gsnr=1., beta_star=1.):
=======
from selection.randomized.lasso_iv import lasso_iv, rescaled_lasso_iv, stat_lasso_iv
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

# Sigma_12 is the true Sigma_{12}
def test_lasso_iv_instance(n=1000, p=10, s=3, ndraw=5000, burnin=5000, Sigma_12=0.8, gsnr=1., beta_star=1.):

    #inst, const = bigaussian_instance, lasso_iv
    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, gsnr=gsnr,beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    #n, p = Z.shape

    conv = lasso_iv(Y, D, Z)
    conv.fit()

    pivot, _, _ = conv.summary(parameter=beta_star)

    return pivot

def test_pivots(nsim=500, n=1000, p=10, s=3, ndraw=5000, burnin=5000, Sigma_12=0.8, gsnr=1., beta_star=1.):
    P0 = []
    for i in range(nsim):
        try:
            p0 = test_lasso_iv_instance(n=n, p=p, s=s, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        except:
            p0 = []
        P0.extend(p0)

    print(np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))

    U = np.linspace(0, 1, 101)
    plt.plot(U, ECDF(P0)(U))
    plt.plot(U, U, 'r--')
    plt.show()

# Sigma_12 is the true Sigma_{12}
def test_rescaled_lasso_iv_instance(n=1000, p=10, s=3, ndraw=5000, burnin=5000, Sigma_12=0.8, gsnr=1., beta_star=1.):

    #inst, const = bigaussian_instance, lasso_iv
    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, gsnr=gsnr,beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    #n, p = Z.shape

    conv = rescaled_lasso_iv(Y, D, Z)
    conv.fit()

    pivot, _, _ = conv.summary(parameter=beta_star)

    return pivot

def test_pivots_rescaled(nsim=500, n=1000, p=10, s=3, ndraw=5000, burnin=5000, Sigma_12=0.8, gsnr=1., beta_star=1.):
    P0 = []
    for i in range(nsim):
        try:
            p0 = test_rescaled_lasso_iv_instance(n=n, p=p, s=s, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        except:
            p0 = []
        P0.extend(p0)

    print(np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))

    U = np.linspace(0, 1, 101)
    plt.plot(U, ECDF(P0)(U))
    plt.plot(U, U, 'r--')
    plt.show()

# Sigma_12 is the true Sigma_{12}
def test_stat_lasso_iv_instance(n=1000, p=10, s=3, ndraw=5000, burnin=5000, Sigma_12=0.8, gsnr=1., beta_star=1.):
>>>>>>> f125cdb72d4da3c41710cf85fa9c797c5b9c0678

    #inst, const = bigaussian_instance, lasso_iv
    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, gsnr=gsnr,beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    #n, p = Z.shape

    conv = stat_lasso_iv(Y, D, Z)
    conv.fit()

<<<<<<< HEAD
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0]) and conv._inactive.sum()>0:
        pivot, _, _ = conv.summary(parameter=beta_star)
    return pivot

def test_pivots_stat(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, Sigma_12=0.8, gsnr=1., beta_star=1.):
=======
    pivot, _, _ = conv.summary(parameter=beta_star)

    return pivot

def test_pivots_stat(nsim=500, n=1000, p=10, s=3, ndraw=5000, burnin=5000, Sigma_12=0.8, gsnr=1., beta_star=1.):
>>>>>>> f125cdb72d4da3c41710cf85fa9c797c5b9c0678
    P0 = []
    for i in range(nsim):
        try:
            p0 = test_stat_lasso_iv_instance(n=n, p=p, s=s, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        except:
            p0 = []
        P0.extend(p0)

    print(np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))

    U = np.linspace(0, 1, 101)
    plt.plot(U, ECDF(P0)(U))
    plt.plot(U, U, 'r--')
    plt.show()


def main(nsim=500):

    P0 = []
    from statsmodels.distributions import ECDF

    n, p, s = 1000, 10, 3
    Sigma_12 = 0.8
    gsnr = 1.
    beta_star = 1.

    for i in range(nsim):
        try:
            p0 = test_lasso_iv_instance(n=n, p=p, s=s, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        except:
            p0 = []
        P0.extend(p0)

    print(np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))

    U = np.linspace(0, 1, 101)
    #plt.clf()
    plt.plot(U, ECDF(P0)(U))
    plt.plot(U, U, 'r--')
    #plt.savefig("plot.pdf")
    plt.show()


if __name__ == "__main__":
    main()
