import numpy as np

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso_iv import lasso_iv
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

# sigma is the true Sigma_{12}
def test_lasso_iv_instance(n=1000, p=10, s=3, ndraw=5000, burnin=5000, sigma=0.8, gsnr=1., beta_star=1.):

    #inst, const = bigaussian_instance, lasso_iv
    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, gsnr=gsnr,beta=beta_star,Sigma=np.array([[1., sigma],[sigma, 1.]]))

    #n, p = Z.shape

    conv = lasso_iv(Y, D, Z)
    conv.fit()

    pivot, _, _ = conv.summary(parameter=beta_star, Sigma=sigma)

    return pivot

def test_pivots(nsim=500, n=1000, p=10, s=3, ndraw=5000, burnin=5000, sigma=0.8, gsnr=1., beta_star=1.):
    P0 = []
    for i in range(nsim):
        try:
            p0 = test_lasso_iv_instance(n=n, p=p, s=s, sigma=sigma, gsnr=gsnr, beta_star=beta_star)
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
    sigma = 0.8
    gsnr = 1.
    beta_star = 1.

    for i in range(nsim):
        try:
            p0 = test_lasso_iv_instance(n=n, p=p, s=s, sigma=sigma, gsnr=gsnr, beta_star=beta_star)
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
