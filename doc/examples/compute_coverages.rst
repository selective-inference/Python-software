Coverage of randomized LASSO intervals
--------------------------------------

In this example, we demonstrate how to compute confidence intervals
for a randomized LASSO example, as well as demonstrating
that the selective pivots are uniformly distributed.

.. nbplot::

    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.distributions import ECDF

.. mpl-interactive

First, we define a function that will fit a randomized LASSO and
return both the pivotal quantites and confidence intervals.

.. nbplot::

    from selection.tests.instance import gaussian_instance
    from selection.randomized.convenience import lasso

    def fit_randomized_LASSO(ndraw=20000, burnin=2000):

    for const_info, rand in product(zip([gaussian_instance], [cls.gaussian]), ['laplace', 'gaussian']):

        X, Y, beta, _, _ = gaussian_instance(n=100, p=20, s=3, sigma=5.)
        n, p = X.shape
        W = np.ones(X.shape[1]) * 8
        L = lasso.gaussian(X, Y, W, randomizer='gaussian', parametric_cov_estimator=True)

        # the active set and signs of the LASSO fit
        signs = conv.fit()

        # for computational efficiency, we marginalize over 
        # inactive coordinates when possible

        marginalizing_groups = np.ones(p, np.bool)
        conv.decompose_subgradient(marginalizing_groups=marginalizing_groups)

        selected_features = conv._view.selection_variable['variables']
        nactive = selected_features.sum()

        if nactive==0:
            return None
        else:
            sel_pivots, sel_ci = L.summary(selected_features,
                                           null_value=beta[selected_features],
                                           ndraw=10000,
                                           burnin=2000,
                                           compute_intervals=True)
            return sel_pivots, sel_ci, beta[selected_features]

Let's do a test run

.. nbplot::

    def compute_coverage(sel_ci, true_vec):
        nactive = true_vec.shape[0]
        coverage = np.zeros(nactive)
        for i in range(nactive):
            if true_vec[i]>=sel_ci[i,0] and true_vec[i]<=sel_ci[i,1]:
                coverage[i]=1
        return coverage


    def main(ndraw=20000, burnin=5000, nsim=50):
        np.random.seed(1)

        sel_pivots_all = list()
        sel_ci_all = list()
        rand_all = []
        for i in range(nsim):
            for idx, results in enumerate(test_opt_weighted_intervals(ndraw=ndraw, burnin=burnin)):
                if results is not None:
                    (rand, sel_pivots, sel_ci, true_vec) = results
                    if i==0:
                        sel_pivots_all.append([])
                        rand_all.append(rand)
                        sel_ci_all.append([])
                    sel_pivots_all[idx].append(sel_pivots)
                    print(sel_ci)
                    sel_ci_all[idx].append(compute_coverage(sel_ci, true_vec))

        xval = np.linspace(0, 1, 200)

        for idx in range(len(rand_all)):
            fig = plt.figure(num=idx, figsize=(8,8))
            plt.clf()
            sel_pivots_all[idx] = [item for sublist in sel_pivots_all[idx] for item in sublist]
            plt.plot(xval, ECDF(sel_pivots_all[idx])(xval), label='selective')
            plt.plot(xval, xval, 'k-', lw=1)
            plt.legend(loc='lower right')

            sel_ci_all[idx] = [item for sublist in sel_ci_all[idx] for item in sublist]
            print(sel_ci_all)
            plt.title(''.join(["coverage ", str(np.mean(sel_ci_all[idx]))]))
            plt.savefig(''.join(["fig", rand_all[idx], '.pdf']))

