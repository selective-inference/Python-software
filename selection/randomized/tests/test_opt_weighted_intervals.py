from itertools import product
import numpy as np
import nose.tools as nt

from selection.randomized.convenience import lasso, step, threshold
from selection.randomized.query import optimization_sampler
from selection.tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from selection.tests.flags import SMALL_SAMPLES
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_iftrue

from scipy.stats import t as tdist
from selection.randomized.glm import target as glm_target, glm_nonparametric_bootstrap, pairs_bootstrap_glm
from selection.randomized.M_estimator import restricted_Mest

@set_seed_iftrue(True, 200)
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=1000, burnin=100)
def test_opt_weighted_intervals(ndraw=20000, burnin=2000):

    cls = lasso
    for const_info, rand in product(zip([gaussian_instance], [cls.gaussian]), ['laplace']):

        inst, const = const_info

        X, Y = inst(n=100, p=10, s=0)[:2]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 1
        conv = const(X, Y, W, randomizer=rand)
        signs = conv.fit()
        print("signs", signs)

        #marginalizing_groups = np.zeros(p, np.bool)
        #marginalizing_groups[:int(p/2)] = True
        #conditioning_groups = ~marginalizing_groups
        #conditioning_groups[-int(p/4):] = False

        selected_features = conv._view.selection_variable['variables']

        #conv.summary(selected_features,
        #             ndraw=ndraw,
        #             burnin=burnin,
        #             compute_intervals=True)

        #conv.decompose_subgradient(marginalizing_groups=marginalizing_groups,
        #                           conditioning_groups=conditioning_groups)

        conv._queries.setup_sampler(form_covariances=None)
        conv._queries.setup_opt_state()
        opt_sampler = optimization_sampler(conv._queries)

        S = opt_sampler.sample(ndraw,
                               burnin,
                               stepsize=1.e-3)
        #print(S.shape)
        #print([np.mean(S[:,i]) for i in range(p)])

        unpenalized_mle = restricted_Mest(conv.loglike, selected_features)
        form_covariances = glm_nonparametric_bootstrap(n, n)
        #conv._queries.setup_sampler(form_covariances)
        boot_target, boot_target_observed = pairs_bootstrap_glm(conv.loglike, selected_features, inactive=None)
        opt_sampler.setup_target(boot_target,
                                 form_covariances)

        selective_pvalues = opt_sampler.coefficient_pvalues(unpenalized_mle, sample=S)
        print("pvalues ", selective_pvalues)
        selective_CI = opt_sampler.confidence_intervals(unpenalized_mle, sample=S)
        print(selective_CI)

        return selective_CI


test_opt_weighted_intervals()