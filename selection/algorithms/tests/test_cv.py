from __future__ import print_function
import numpy as np

from selection.tests.instance import gaussian_instance 
from selection.algorithms.cross_valid import lasso_tuned, lasso_tuned_conditional 
from selection.distributions.discrete_family import discrete_family

def test_CV(ndraw=500, sigma_known=True,
            burnin=100,
            s=7,
            rho=0.3,
            method=lasso_tuned,
            snr=5):
    # generate a null and alternative pvalue
    # from a particular model

    X, Y, beta, active, sigma = gaussian_instance(n=500, p=100, s=s, rho=rho, snr=snr)
    if sigma_known:
        sigma = sigma
    else:
        sigma = None

    method_ = method(Y, X, scale_inter=0.0001, scale_valid=0.0001, scale_select=0.0001)

    if True: 
        do_null = True
        if do_null:
            which_var = method_.active_set[s] # the first null one
            method_.setup_inference(which_var) ; iter(method_)

            for i in range(ndraw + burnin):
                method_.next()

            Z = np.array(method_.null_sample[which_var][burnin:])
            family = discrete_family(Z, 
                                     np.ones_like(Z))
            obs = method_._gaussian_obs[which_var]

            pval0 = family.cdf(0, obs)
            pval0 = 2 * min(pval0, 1 - pval0)
        else:
            pval0 = np.random.sample()

        which_var = 0
        method_.setup_inference(which_var); iter(method_)
        for i in range(ndraw + burnin):
            method_.next()

        family = discrete_family(method_.null_sample[which_var][burnin:], 
                                 np.ones(ndraw))
        obs = method_._gaussian_obs[which_var]
        pvalA = family.cdf(0, obs)
        pvalA = 2 * min(pvalA, 1 - pvalA)
        return pval0, pvalA, method_

def plot_fig():

    from statsmodels.distributions import ECDF
    import matplotlib.pyplot as plt
    f = plt.figure(num=1)

    s = 7
    P0, PA = [], []
    screened = 0

    results = {}
    counter = {}
    linestyle = {lasso_tuned:'-',
                 lasso_tuned_conditional:'-.'}

    results.setdefault('indep', [])

    for i in range(200):
        print(i)
        for method in [lasso_tuned, lasso_tuned_conditional]:
            result = test_CV(ndraw=1000, burnin=500, sigma_known=False,
                              method=method, s=s)
            counter.setdefault(method, 0) 
            if result is not None:
                results.setdefault(method, []).append(result[:2])
                counter[method] += 1

                P0 = np.array(results[method])[:,0]
                PA = np.array(results[method])[:,1]

                U = np.linspace(0,1,101)
                ecdf0 = ECDF(P0)(U)
                ecdfA = ECDF(PA)(U)
                ax = f.gca()
                ax.plot(U, ecdf0, 'k' + linestyle[method], 
                        linewidth=3,
                        label=str(method.__name__)[11:])
                ax.plot(U, ecdfA, 'r' + linestyle[method], 
                        linewidth=3)
                results['indep'].append((result[2].pval_indep[s], result[2].pval_indep[0]))
                np.savez(str(method.__name__)[11:] + '.npz', P0=P0, PA=PA)

            print(('screening', str(method.__name__)), (counter[method] * 1.) / (i + 1))
            print(('power', str(method.__name__)), np.mean(PA < 0.05))
            print(('level', str(method.__name__)), np.mean(P0 < 0.05))

        P0 = np.array(results['indep'])[:,0]
        PA = np.array(results['indep'])[:,1]
        np.savez('indep.npz', P0=P0, PA=PA)

        print(('power', 'indep'), np.mean(PA < 0.05))
        print(('level', 'level'), np.mean(P0 < 0.05))


        U = np.linspace(0,1,101)
        ecdf0 = ECDF(P0)(U)
        ecdfA = ECDF(PA)(U)

        ax.plot(U, ecdf0, 'k:',
                linewidth=3,
                label='independent')
        ax.plot(U, ecdfA, 'r:',
                linewidth=3)

        ax.legend(loc='lower right')
        f.savefig('ecdf.pdf')
        f.clf()


