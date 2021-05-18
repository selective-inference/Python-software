import numpy as np

from ..lasso import lasso, selected_targets
from ...tests.instance import gaussian_instance

def UMVU(query,
         X,
         Y,
         nonzero,
         feat,
         dispersion):

    n, p = X.shape

    nopt = nonzero.sum()

    _, randomizer_prec = query.randomizer.cov_prec

    implied_precision = np.zeros((n + nopt, n + nopt))

    implied_precision[:n][:, :n] = (1. / dispersion) * (np.identity(n)) + (X.dot(X.T) * randomizer_prec)

    implied_precision[n:][:, :n] = -query.opt_linear.T.dot(X.T) * randomizer_prec

    implied_precision[:n][:, n:] = implied_precision[n:][:, :n].T

    implied_precision[n:][:, n:] = query.opt_linear.T.dot(query.opt_linear) * randomizer_prec

    implied_cov = np.linalg.inv(implied_precision)

    _prec = np.linalg.inv(implied_cov[:n][:, :n])

    linear_coef = (np.linalg.pinv(X[:, feat]).dot(_prec))
    offset = -np.linalg.pinv(X[:, feat]).dot(X.dot(query.initial_subgrad)
                                             - _prec.dot(implied_cov[:n][:, n:]).dot(query.opt_linear.T.dot(query.initial_subgrad))) * (randomizer_prec)

    linear_coef *= dispersion
    offset *= dispersion
    UMVU = linear_coef.dot(Y) + offset

    return UMVU

def EST(query,
        X,
        Y,
        nonzero,
        feat,
        dispersion):

    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(query.loglike,
                                      query._W,
                                      feat,
                                      dispersion=dispersion)

    _, randomizer_prec = query.randomizer.cov_prec
    cond_cov = query.cond_cov
    logdens_linear = query.sampler.logdens_transform[0]
    cond_mean = query.cond_mean

    prec_target = np.linalg.inv(cov_target)
    prec_opt = np.linalg.inv(cond_cov)

    target_linear = cov_target_score.T.dot(prec_target)
    target_offset = (-X.T.dot(Y) + query.initial_subgrad) - target_linear.dot(observed_target)

    target_lin = - logdens_linear.dot(target_linear)
    target_off = cond_mean - target_lin.dot(observed_target)

    _prec = prec_target + (target_linear.T.dot(target_linear) * randomizer_prec) - target_lin.T.dot(
        prec_opt).dot(target_lin)
    _P = target_linear.T.dot(target_offset) * randomizer_prec

    linear_coef = cov_target.dot(_prec)
    offset = cov_target.dot(_P - target_lin.T.dot(prec_opt).dot(target_off))
    est = linear_coef.dot(observed_target) + offset

    return est

def test_UMVU(n=500,
              p=100,
              signal_fac=1.,
              s=5,
              sigma=3.,
              rho=0.7,
              randomizer_scale=np.sqrt(0.5)):


    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=True,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        sigma_ = np.std(Y)
        W = 0.8 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     #ridge_term=0.,
                     randomizer_scale=randomizer_scale * sigma)

        signs = conv.fit()
        nonzero = signs != 0

        if nonzero.sum() > 0:
            #dispersion = sigma ** 2
            if p > n/2:
                dispersion = np.std(Y) ** 2
            else:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

            feat = nonzero.copy()
            feat[-5:] = 1
            dispersion = np.linalg.norm(Y - X[:, feat].dot(np.linalg.pinv(X[:, feat]).dot(Y))) ** 2 / (n - feat.sum())

            umvu = UMVU(conv,
                        X,
                        Y,
                        nonzero,
                        feat,
                        dispersion)

            est = EST(conv,
                      X,
                      Y,
                      nonzero,
                      feat,
                      dispersion)

            print("check ", np.allclose(est, umvu, atol=1e-04), umvu, est)

            return umvu, est

def main():

    test_UMVU(n=400, p=100, s=5)

if __name__ == "__main__":
    main()
