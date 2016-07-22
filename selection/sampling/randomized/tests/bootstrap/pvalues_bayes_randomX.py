import numpy as np
from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist, percentileofscore
from selection.sampling.langevin import projected_langevin

def pval(vec_state, full_projection,
         X, obs_residuals, beta_unpenalized, full_null, signs, lam, epsilon,
         nonzero, active,
         Sigma,
         weights, randomization_dist,
         Langevin_steps, step_size, burning):
    """
    """

    n, p = X.shape

    null = []
    alt = []

    X_E = X[:, active]
    inactive = ~active
    nalpha = n
    nactive = np.sum(active)
    ninactive  = np.sum(inactive)

    active_set = np.where(active)[0]

    print "true nonzero ", nonzero, "active set", active_set

    XEpinv = np.linalg.pinv(X[:, active])
    hessian = np.dot(X.T, X)
    hessian_reistricted = hessian[:,active]
    mat = XEpinv.dot(np.diag(obs_residuals))


    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):
            if j>0:
                break

            eta = np.zeros(nactive)
            eta[j] = 1
            sigma_eta_sq = Sigma[j,j]

            linear_part = np.identity(nactive) - (np.outer(np.dot(Sigma, eta), eta) / sigma_eta_sq)
            #P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
            #T_minus_j = np.dot(P, beta_unpenalized)
            T_minus_j = np.dot(linear_part, beta_unpenalized) # sufficient stat for the nuisance
            c = np.dot(Sigma, eta) / sigma_eta_sq
            fixed_part = full_null + hessian_reistricted.dot(T_minus_j)

            XXc = hessian_reistricted.dot(c)

            def full_gradient(vec_state, fixed_part=fixed_part, obs_residuals=obs_residuals,
                              eta = eta,
                              lam=lam, epsilon=epsilon, active=active, inactive=inactive):

                nactive = np.sum(active)
                ninactive = np.sum(inactive)

                alpha = vec_state[:n]
                betaE = vec_state[n:(n + nactive)]
                cube = vec_state[(n + nactive):]

                beta_full = np.zeros(p)
                beta_full[active] = betaE
                subgradient = np.zeros(p)
                subgradient[inactive] = lam * cube
                subgradient[active] = lam * signs

                opt_vec = epsilon * beta_full + subgradient

                beta_bar_j_boot = np.inner(mat[j,:],alpha)
                omega = - fixed_part - XXc * beta_bar_j_boot + np.dot(hessian_reistricted, betaE) + opt_vec

                if randomization_dist=="laplace":
                    randomization_derivative = np.sign(omega)  # sign(w), w=grad+\epsilon*beta+lambda*u
                if randomization_dist=="logistic":
                    randomization_derivative = -(np.exp(-omega)-1)/(np.exp(-omega)+1)
                A = hessian + epsilon * np.identity(nactive + ninactive)
                A_restricted = A[:, active]

                _gradient = np.zeros(n + nactive + ninactive)

                # saturated model
                mat_q = np.outer(XXc, eta).dot(mat)

                _gradient[:n] = np.dot(mat_q.T, randomization_derivative)

                if (weights == 'exponential'):
                    _gradient[:n] -= np.ones(n)
                if (weights=="normal"):
                    _gradient[:n] -= alpha
                if weights == "gamma":
                    _gradient[:n] = 3./(alpha+2)-2
                if (weights == "gumbel"):
                       gumbel_beta = np.sqrt(6)/(1.14*np.pi)
                       euler = 0.57721
                       gumbel_mu = -gumbel_beta * euler
                       gumbel_sigma = 1./1.14
                       _gradient[:n] -= (1.-np.exp(-(alpha*gumbel_sigma-gumbel_mu)/gumbel_beta))*gumbel_sigma/gumbel_beta
                _gradient[n:(n + nactive)] = - A_restricted.T.dot(randomization_derivative)
                _gradient[(n + nactive):] = - lam * randomization_derivative[inactive]

                # selected model
                # _gradient[:nactive] = - (np.dot(Sigma_T_inv, data[:nactive]) + np.dot(hessian[:, active].T, sign_vec))
                # _gradient[ndata:(ndata + nactive)] = np.dot(A_restricted.T, sign_vec)
                # _gradient[(ndata + nactive):] = lam * sign_vec[inactive]

                return _gradient

            sampler = projected_langevin(vec_state.copy(),
                                         full_gradient,
                                         full_projection,
                                         step_size)

            samples = []


            for i in range(Langevin_steps):
                sampler.next()
                if (i>burning):
                    samples.append(sampler.state.copy())

            samples = np.array(samples)
            alpha_samples = samples[:, :n]

            beta_bars = [np.dot(XEpinv, np.diag(obs_residuals)).dot(alpha_samples[i,:].T) for i in range(len(samples))]


            pop = [z[j] for z in beta_bars]
            obs = beta_unpenalized[j]

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
