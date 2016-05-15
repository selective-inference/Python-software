import numpy as np
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist, percentileofscore

def pval_new(sampler,
         loss_args,
         linear_part, # not used anymore
         data,
         nonzero,
         Sigma,
         true_beta):
    """
    The function computes the null and alternative pvalues for a regularized problem.
    The null p-values correspond to the coefficients whose corresponding predictors
    are in the active set but not in the true support (true support denoted as nonzero in the code)
    and the alternative are the ones whose corresponding predictors are in the true support.

    Parameters:
    -----------
    to be added

    nonzero: the true underlying nonzero pattern (true support)

    sigma: noise level of the data, if "None",
    estimates of covariance is needed

    Returns:
    --------
    null, alt: null and alternative pvalues.
    """

    n, p = sampler.loss.X.shape

    data0 = data.copy()
    size_data = data0.shape[0]

    active = sampler.penalty.active_set # E

    size_active = np.sum(active) ## |E|, size of the active set
    #nnuisance = size_active-1 ## number of nuisance parameters, assuming we are in the selected model it becomes |E|-1
                              ## if we are testing for \beta_j, all \beta_{E\setminus j} are nuisance parameters

    if linear_part is None: # this part haven't read
        off = ~np.identity(p, dtype=bool)
        E = np.zeros((p,p), dtype=bool)
        E[off] = active
        E = np.logical_or(E.T, E)
        active_set = np.where(E[off])[0]
    else:
        active_set = np.where(active)[0]

    print "true nonzero ", nonzero, "nonzero coefs", active_set

    null = [] # corresponding to tests H_0: beta_j^E=0, the predictor corresponding to beta_j (j=1,...,|E|)
              # (idx column of X below) is not in the true support
    alt = []  # corresponding to tests H_0: beta_j^E=0, the predictor  corresponding to beta_j (j=1,....|E|)
              # (idx column of X below) is in the true support


    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):  # testing H0: \beta_j^E=0, j=1,..., |E|
                                              # \beta_j^E corresponds to the predictor X[:,idx] in the model X[:,E]
            if linear_part is not None:
                eta = np.zeros(size_active)
                eta[j] = 1    # eta = e_j \in R^{|E|}

                sigma_eta_sq = np.dot(np.dot(eta.T, Sigma), eta)    # \eta^T \Sigma \eta, for the eta above this should be Sigma[j,j]
                # print 'sigma_eta', sigma_eta
                # print 'sigma_jj', Sigma[j,j]

                # L1=(I-\frac{\Sigma\eta\eta^T}{\sigma_eta^2}), conditioning is on L1*\bar{\beta}_E
                L1 = np.identity(size_active) - (np.outer(np.dot(Sigma, eta), eta)/sigma_eta_sq)

                #keep = np.copy(active)
                #keep[idx] = False

                # adding |E|\times(p-|E|) matrix of zero on L1 to get L, then conditioning done on L * data fixed
                # for the selected model
                L2 = np.concatenate((L1, np.zeros((size_active, size_data-size_active))), axis=1)

                # for the saturated model
                L3 = np.concatenate((np.zeros((size_data-size_active, size_active )), np.identity(size_data-size_active) ), axis=1)
                L = np.concatenate((L2, L3), axis=0)
                # print L
                # print L.shape

                #print L[:,j]
                #print L[:, 3]
                eta1 = np.concatenate((eta, np.zeros(p-size_active)), axis=0)

                #L = linear_part[:,keep]

                loss_args['linear_part'] = L
                loss_args['value'] = np.dot(L, data)  # conditioning is on L*data=value
                loss_args['beta'] = true_beta.copy()
                #print 'true_beta', true_beta
                loss_args['beta'][j]=0

                #print 'loss_args',loss_args['beta']

                sampler.setup_sampling(data, loss_args=loss_args)
                samples = sampler.sampling(ndraw=5000, burnin=1000)
                pop = [np.dot(eta1,z) for z, _, in samples]
                obs = np.dot(eta1, data0)   # observed \bar{\beta}_{E,j}
            else:   #this part haven't read
                row, col = nonzero_index(idx, p)
                print row, col
                eta = data0[:, row]
                sampler.setup_sampling(data, loss_args=loss_args)
                samples = sampler.sampling(ndraw=5000, burnin=1000)
                pop = [np.dot(eta1, z[:, col]) for z, _, in samples]
                obs = np.dot(eta1, data0[:, col])


            fam = discrete_family(pop, np.ones_like(pop))
            pval = fam.cdf(0, obs)
            pval = 2 * min(pval, 1-pval)
            print "observed: ", obs, "p value: ", pval
            if pval < 0.0001:
                print obs, pval, np.percentile(pop, [0.2,0.4,0.6,0.8,1.0])
            if idx in nonzero:   # H0:\beta_j^E=0 is false if idx in the true support
                alt.append(pval)
            else:                # H0:\beta_j^E=0 is true if idx is not in the true support
                null.append(pval)

            # prints the counts of MCMC movements out of the total ndraw+burnin
            print 'opt_vars', sampler.penalty.accept_l1_part, sampler.penalty.total_l1_part
            print 'data', sampler.loss.accept_data, sampler.loss.total_data

    return null, alt


def nonzero_index(idx, p): #haven't read this
    off = ~np.identity(p, dtype=bool)
    M = np.full((p, p), np.nan)
    M[off] = np.arange(p*(p-1))
    loc = np.where(M == idx)
    return loc[0][0], loc[1][0]
