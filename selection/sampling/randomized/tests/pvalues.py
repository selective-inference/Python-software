import numpy as np
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist, percentileofscore

def pval(sampler, 
         loss_args,
         linear_part,
         data, y,
         nonzero):
    """
    The function computes the null and alternative 
    pvalues for a regularized problem. The null p-values correspond to
    the coefficients whose corresponding predictors are in the active set
    but not in the true support (denoted as nonzero in the code) and the
    alternative are the ones whose corresponding predictors are in the true support.

    Parameters:
    -----------
    loss: specific loss, 
    e.g. gaussian_Xfixed, logistic_Xrandom

    penalty: regularization, e.g. selective_l1norm

    randomization: the distribution of the randomized noise

    linear part, data: (C, y)
    To test for the jth parameter, we condition on the 
    C_{\backslash j} y = d_{\backslash j}.

    nonzero: the true underlying nonzero pattern (true support)

    sigma: noise level of the data, if "None", 
    estimates of covariance is needed

    Returns:
    --------
    null, alt: null and alternative pvalues.
    """
    
    n, p = sampler.loss.X.shape

    data0 = data.copy()

    active = sampler.penalty.active_set


    if linear_part is None:
        off = ~np.identity(p, dtype=bool)
        E = np.zeros((p,p), dtype=bool)
        E[off] = active
        E = np.logical_or(E.T, E)
        active_set = np.where(E[off])[0]
    else:
        active_set = np.where(active)[0]

    print "true nonzero ", nonzero, "nonzero coefs", active_set

    null = []
    alt = []

    X = linear_part.copy()
    X_E = X[:, active]
    #print 'active_set', X[1,active_set]
    #print 'active', X[1, active]

    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):
            if linear_part is not None:
                e_j = np.zeros(X_E.shape[1])
                e_j[j] = 1
                eta = linear_part[:,idx] 
                keep = np.copy(active)
                # keep=np.arange(X.shape[1])

                keep[idx] = False
                L = linear_part[:,keep]

                loss_args['linear_part'] = L.T
                loss_args['value'] = np.dot(L.T, data)
                sampler.setup_sampling(data, loss_args=loss_args)
                samples = sampler.sampling(ndraw=5000, burnin=1000)
                R = sampler.loss.R
                P = sampler.loss.P

                #pop = [np.dot(eta, z) for z, _, in samples]
                #pop = [np.dot(eta, np.dot(P,y)+np.dot(R,z)) for z, _, in samples]
                #obs = np.dot(eta, data0)
                pop = [np.dot(e_j, np.linalg.lstsq(X_E, np.dot(P,y)+np.dot(R,z))[0]) for z, _, in samples]
                #pop = [np.dot(e_j, np.linalg.lstsq(X_E, z)[0]) for z, _, in samples]
                obs = np.dot(e_j, np.linalg.lstsq(X_E, y)[0])
                #print 'obs', obs
            else:
                row, col = nonzero_index(idx, p)
                print row, col
                eta = data0[:, row] 
                sampler.setup_sampling(data, loss_args=loss_args)
                samples = sampler.sampling(ndraw=5000, burnin=1000)

                pop = [np.dot(eta, z[:, col]) for z, _, in samples]
                obs = np.dot(eta, data0[:, col])


            fam = discrete_family(pop, np.ones_like(pop))
            pval = fam.cdf(0, obs)
            pval = 2 * min(pval, 1-pval)
            print "observed: ", obs, "p value: ", pval
            if pval < 0.0001:
                print obs, pval, np.percentile(pop, [0.2,0.4,0.6,0.8,1.0]) 
            if idx in nonzero:
                alt.append(pval)
            else:
                null.append(pval)
            print 'opt_vars', sampler.penalty.accept_l1_part, sampler.penalty.total_l1_part
            print 'data', sampler.loss.accept_data, sampler.loss.total_data

    return null, alt

def nonzero_index(idx, p):
    off = ~np.identity(p, dtype=bool)
    M = np.full((p, p), np.nan)
    M[off] = np.arange(p*(p-1)) 
    loc = np.where(M == idx)
    return loc[0][0], loc[1][0]
