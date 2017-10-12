
Coverage of randomized LASSO intervals
--------------------------------------

In this example, we demonstrate how to compute confidence intervals for
a randomized LASSO example, as well as demonstrating that the selective
pivots are uniformly distributed.

.. nbplot::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from statsmodels.distributions import ECDF

.. raw:: html

   <!-- mpl-interactive -->

First, we define a function that will fit a randomized LASSO and return
both the pivotal quantites and confidence intervals. The design matrix
is equicorrelated with parameter :math:`\rho=0.2`.

.. nbplot::

    >>> from selection.tests.instance import gaussian_instance
    >>> from selection.randomized.convenience import lasso
    >>>
    >>> def fit_randomized_LASSO(ndraw=10000, burnin=2000, marginalize=False):
    ...
    ...     X, Y, beta, true_active, _ = gaussian_instance(n=100, p=20, s=3, sigma=5., signal=5)
    ...     n, p = X.shape
    ...     W = np.ones(X.shape[1]) * 30
    ...     L = lasso.gaussian(X, Y, W, randomizer='gaussian', parametric_cov_estimator=True)
    ...
    ...     # the active set and signs of the LASSO fit
    ...     signs = L.fit()
    ...
    ...     # for computational efficiency, we can 
    ...     # marginalize over inactive coordinates 
    ...
    ...     if marginalize:
    ...         marginalizing_groups = np.ones(p, np.bool)
    ...         L.decompose_subgradient(marginalizing_groups=marginalizing_groups)
    ...
    ...     selected_features = signs != 0
    ...     nactive = selected_features.sum()
    ...
    ...     if set(np.nonzero(selected_features)[0]).issuperset(true_active):
    ...         sel_pivots, sel_pval, sel_ci = L.summary(selected_features,
    ...                                                  parameter=beta[selected_features],
    ...                                                  ndraw=ndraw,
    ...                                                  burnin=burnin,
    ...                                                  compute_intervals=True)
    ...
    ...         return sel_pivots, sel_pval, sel_ci, beta[selected_features]

Let’s do a test run

.. nbplot::

    >>> fit_randomized_LASSO()
    (array([ 0.43548428,  0.03278839,  0.00481199]),
     array([ 0.        ,  0.        ,  0.97660498]),
     array([[ 18.97524697,  40.49266138],
            [ 28.08291483,  48.76959338],
            [-12.15053136,  14.24711888]]),
     array([ 25.,  25.,  25.]))

.. nbplot::

    >>> def compute_coverage(sel_ci, truth):
    ...     coverage = (sel_ci[:,0] <= truth) * (sel_ci[:,1] >= truth)
    ...     return coverage

.. nbplot::

    >>> def main(ndraw=10000, burnin=2000, nsim=50):
    ...     np.random.seed(1)
    ...
    ...     sel_pivots_all = []
    ...     P0 = []
    ...     PA = []
    ...     sel_coverage = []
    ...
    ...     for i in range(nsim):
    ...         results = fit_randomized_LASSO(ndraw=ndraw, burnin=burnin)
    ...         if results is not None:
    ...             sel_pivots, sel_pval, sel_ci, truth = results
    ...             P0.extend(sel_pval[truth == 0])
    ...             PA.extend(sel_pval[truth != 0])
    ...             sel_pivots_all.extend(sel_pivots)
    ...             sel_coverage.extend(compute_coverage(sel_ci, truth))
    ...
    ...     return sel_pivots_all, sel_coverage, P0, PA

Make a plot
~~~~~~~~~~~

.. nbplot::

    >>> sel_pivots_all, sel_coverage, P0, PA = main(nsim=30)
    >>> xval = np.linspace(0, 1, 200)

.. mpl-interactive::

.. nbplot::

    >>> fig = plt.figure(figsize=(8,8))
    >>> plt.plot(xval, ECDF(sel_pivots_all)(xval), label='Pivot')
    >>> plt.plot(xval, ECDF(P0)(xval), label='H0')
    >>> plt.plot(xval, ECDF(PA)(xval), label='HA')
    >>>
    >>> plt.plot(xval, xval, 'k-', lw=1)
    >>> plt.legend(loc='lower right')
    <...>



What does our coverage look like?

.. nbplot::

    >>> print(np.mean(sel_coverage))

    0.876033057851


