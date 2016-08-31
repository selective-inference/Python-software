import numpy as np
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist, percentileofscore
from selection.sampling.langevin import projected_langevin

def pval(vec_state0, full_projection,
         X, y, epsilon, lam,
         nonzero, active,
         beta_reference,
         Langevin_steps, burning, step_size, randomization_distribution):
    """
    """

    n, p = X.shape
    y0 = y.copy()

    null = []
    alt = []

    XE = X[:, active]
    XEpinv = np.linalg.pinv(XE)
    XEinv = np.linalg.inv(np.dot(XE.T, XE))
    ndata = y.shape[0]
    hessian = np.dot(X.T,X)
    active_set = np.where(active)[0]
    nactive = np.sum(active)
    signs = np.sign(vec_state0[1:(1+nactive)])

    print "true nonzero ", nonzero, "active set", active_set

    all_samples = np.zeros((nactive, Langevin_steps-burning-1))
    all_observed = np.zeros(nactive)
    all_variances = np.zeros(nactive)

    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):
            eta = XEpinv[j,:]
            sigma_sq_eta = np.linalg.norm(eta)**2

            c = np.true_divide(eta, sigma_sq_eta)
            XTc = np.dot(X.T, c)
            fixed_part = np.dot(X.T, np.dot(np.identity(n) - np.outer(c, eta), y0))

            vec_state = vec_state0.copy()
            vec_state[0] = np.inner(eta, y0)

            def full_gradient(vec_state, X=X,
                              lam=lam, epsilon=epsilon, active=active):

                inactive = ~active
                nactive = np.sum(active); ninactive = np.sum(inactive)

                data = vec_state[0]
                betaE = vec_state[1:(1 + nactive)]
                cube = vec_state[(1 + nactive):]

                beta_full = np.zeros(p)
                beta_full[active] = betaE
                subgradient = np.zeros(p)
                subgradient[inactive] = lam * cube
                subgradient[active] = lam * signs

                opt_vec = epsilon * beta_full + subgradient
                gradient = -fixed_part-np.dot(X.T, c*data) + np.dot(hessian, beta_full)
                omega = gradient + opt_vec

                if randomization_distribution == "laplace":
                    randomization_derivative = - np.sign(omega)  # sign(w), w=grad+\epsilon*beta+lambda*u
                if randomization_distribution == "normal":
                    randomization_derivative = - omega
                if randomization_distribution == "logistic":
                    randomization_derivative = (np.exp(-omega) - 1) / (np.exp(-omega) + 1)

                B = hessian + epsilon * np.identity(nactive + ninactive)
                A = B[:, active]

                _gradient = np.zeros(1 + nactive + ninactive)
                _gradient[0] = - (data-beta_reference[j])/sigma_sq_eta - np.inner(XTc, randomization_derivative)
                _gradient[1:(1 + nactive)] = np.dot(A.T, randomization_derivative)
                _gradient[(1 + nactive):] = lam * randomization_derivative[inactive]

                return _gradient


            sampler = projected_langevin(vec_state.copy(),
                                         full_gradient,
                                         full_projection,
                                         step_size)

            samples = []

            for i in range(Langevin_steps):
                if (i>burning):
                    sampler.next()
                    samples.append(sampler.state.copy())

            samples = np.array(samples)
            pop = samples[:, 0]
            obs = np.dot(eta, y0)

            all_samples[j, :] = pop
            all_observed[j] = obs
            all_variances[j] = sigma_sq_eta

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


    return null, alt, all_observed, all_variances, all_samples
