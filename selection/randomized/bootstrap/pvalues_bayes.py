import numpy as np
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist, percentileofscore
from selection.sampling.langevin import projected_langevin

def pval(vec_state, full_projection,
         X, y, obs_residuals, signs, lam, epsilon,
         nonzero, active):
    """
    """

    n, p = X.shape

    y0 = y.copy()

    null = []
    alt = []

    X_E = X[:, active]
    ndata = y.shape[0]
    inactive = ~active
    nalpha = n

    active_set = np.where(active)[0]

    print "true nonzero ", nonzero, "active set", active_set

    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):
            eta = X[:, idx]
            keep = np.copy(active)
            #keep = np.ones(p, dtype=bool)
            keep[idx] = False

            linear_part = X[:,keep].T

            P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
            I = np.identity(linear_part.shape[1])
            R = I - P

            fixed_part = np.dot(X.T, np.dot(P, y))
            hessian = np.dot(X.T, X)
            B = hessian + epsilon * np.identity(p)
            A = B[:, active]

            matXTR = X.T.dot(R)
            def full_gradient(vec_state, fixed_part = fixed_part, R=R,  obs_residuals=obs_residuals, signs=signs,
                              X=X, lam=lam, epsilon=epsilon, data0=y, hessian =hessian, A=A, matXTR= matXTR,
                              nalpha=nalpha, active=active, inactive=inactive):

                nactive = np.sum(active);
                ninactive = np.sum(inactive)

                alpha = vec_state[:nalpha]
                betaE = vec_state[nalpha:(nalpha + nactive)]
                cube = vec_state[(nalpha + nactive):]

                p = X.shape[1]
                beta_full = np.zeros(p)
                beta_full[active] = betaE
                subgradient = np.zeros(p)
                subgradient[inactive] = lam * cube
                subgradient[active] = lam * signs

                opt_vec = epsilon * beta_full + subgradient

                # omega = -  np.dot(X.T, np.diag(obs_residuals).dot(alpha))/np.sum(alpha) + np.dot(hessian, beta_full) + opt_vec
                weighted_residuals = np.diag(obs_residuals).dot(alpha)
                omega = - fixed_part - np.dot(matXTR, weighted_residuals) + np.dot(hessian, beta_full) + opt_vec
                sign_vec = np.sign(omega)

                #mat = np.dot(X.T, np.diag(obs_residuals))
                mat = np.dot(matXTR, np.diag(obs_residuals))
                _gradient = np.zeros(nalpha + nactive + ninactive)
                _gradient[:nalpha] = - np.ones(nalpha) + np.dot(mat.T, sign_vec)
                _gradient[nalpha:(nalpha + nactive)] = - np.dot(A.T, sign_vec)
                _gradient[(nalpha + nactive):] = - lam * sign_vec[inactive]

                return _gradient


            sampler = projected_langevin(vec_state.copy(),
                                         full_gradient,
                                         full_projection,
                                         1. / p)

            samples = []


            for _ in range(5000):
                sampler.next()
                samples.append(sampler.state.copy())

            samples = np.array(samples)
            alpha_samples = samples[:, :n]

            residuals_samples = [np.diag(obs_residuals).dot(alpha_samples[i,:]) for i in range(len(samples))]

            pop = [np.inner(eta, np.dot(P,y0)+np.dot(R,z)) for z in residuals_samples]
            obs = np.inner(eta, y0)

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
