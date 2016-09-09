import numpy as np
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist, percentileofscore
from selection.sampling.langevin import projected_langevin

def pval(vec_state, full_gradient, full_projection, move_data, bootstrap_samples,
         X, y,
         nonzero, active):
    """
    """
    
    n, p = X.shape

    y0 = y.copy()

    null = []
    alt = []

    X_E = X[:, active]
    ndata = y.shape[0]

    active_set = np.where(active)[0]

    print "true nonzero ", nonzero, "active set", active_set

    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):
            eta = X[:, idx]
            #keep = np.copy(active)
            keep = np.ones(p, dtype=bool)
            keep[idx] = False

            linear_part = X[:,keep].T

            P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
            I = np.identity(linear_part.shape[1])
            R = I - P

            sampler = projected_langevin(vec_state.copy(),
                                         full_gradient,
                                         full_projection,
                                         1. / (2*p))

            samples = []

            boot_samples = bootstrap_samples(y0, P, R)

            for _ in range(1000):
                sampler.next()
                new_data = move_data(sampler.state, boot_samples)
                sampler.state[:ndata] = new_data
                samples.append(sampler.state.copy())

            samples = np.array(samples)
            data_samples = samples[:, :n]

            pop = [np.dot(eta,z) for z in data_samples]
            obs = np.dot(eta, y0)


            fam = discrete_family(pop, np.ones_like(pop))
            pval = fam.cdf(0, obs)
            pval = 2 * min(pval, 1-pval)
            print "observed: ", obs, "p value: ", pval
            #if pval < 0.0001:
            #    print obs, pval, np.percentile(pop, [0.2,0.4,0.6,0.8,1.0])
            if idx in nonzero:
                alt.append(pval)
            else:
                null.append(pval)


    return null, alt
