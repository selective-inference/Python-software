from itertools import product
import numpy as np
import nose.tools as nt

from ..convenience import lasso, step, threshold
from ..query import optimization_sampler
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from ...tests.flags import SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue 
from ..glm import glm_nonparametric_bootstrap, pairs_bootstrap_glm

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_optimization_sampler(ndraw=1000, burnin=200):

    cls = lasso
    for const_info, rand, marginalize, condition in product(zip([gaussian_instance,
                                                                 logistic_instance,
                                                                 poisson_instance],
                                                                [cls.gaussian,
                                                                 cls.logistic,
                                                                 cls.poisson]),
                                                                ['gaussian', 'logistic', 'laplace'],
                                                                [False, True],
                                                                [False, True]):

        inst, const = const_info
        X, Y = inst(signal=0.01)[:2]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 80
        conv = const(X, Y, W, randomizer=rand)
        signs = conv.fit()

        if marginalize:
            marginalizing_groups = np.zeros(p, np.bool)
            marginalizing_groups[:int(p/2)] = True
        else:
            marginalizing_groups = None

        if condition:
            if marginalizing_groups is not None:
                conditioning_groups = ~marginalizing_groups
            else:
                conditioning_groups = np.ones(p, np.bool)
            conditioning_groups[-int(p/4):] = False
        else:
            conditioning_groups = None

        selected_features = np.zeros(p, np.bool)
        selected_features[:3] = True

        print(const_info, condition, marginalize, rand)

        conv.decompose_subgradient(conditioning_groups, marginalizing_groups)

        form_covariances = glm_nonparametric_bootstrap(n, n)
        boot_target, boot_target_observed = pairs_bootstrap_glm(conv.loglike, selected_features, inactive=None)
        target_info = boot_target

        opt_samplers = []
        for q in conv._queries.objectives:
            cov_info = q.setup_sampler()
            target_cov, score_cov = form_covariances(target_info,  
                                                     cross_terms=[cov_info],
                                                     nsample=q.nboot)

            opt_samplers.append(q.sampler)

        for opt_sampler in opt_samplers:
            S = opt_sampler.sample(ndraw,
                                   burnin,
                                   stepsize=1.e-10)

        
        
