import numpy as np
from scipy.stats import norm as ndist, t as tdist


def instance(n=200, p=20, s=0, sigma=1, rho=0, snr=7,
             random_signs=False, df=np.inf,
             scale=True, center=True, noise = "normal"):
    """
    A testing instance for the LASSO.
    Design is equi-correlated in the population,
    normalized to have columns of norm 1.

    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    sigma : float
        Noise level

    rho : float
        Equicorrelation value (must be in interval [0,1])

    snr : float
        Size of each coefficient

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    """

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
         np.sqrt(rho) * np.random.standard_normal(n)[:, None])
    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None, :] * np.sqrt(n))
    beta = np.zeros(p)
    beta[:s] = snr
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    active = np.zeros(p, np.bool)
    active[:s] = True

    # noise model

    def _noise_new(n):
        if noise == "normal":
            return np.random.standard_normal(n)
        if noise == "uniform":
            return np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=n)
        if noise == "laplace":
            return np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=n)
        if noise == "logistic":
            return np.random.logistic(loc=0, scale=np.sqrt(3)/np.pi, size=n)



    #def _noise(n, df=np.inf):
    #    if df == np.inf:
    #        return np.random.standard_normal(n)
    #    else:
    #        sd_t = np.std(tdist.rvs(df, size=50000))
    #        return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise_new(n)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma


def bootstrap_covariance(X, y, active, beta_unpenalized):
    n, p = X.shape
    inactive = ~active
    nsample = 5000
    nactive = np.sum(active)

    _mean_cum_data = 0
    _cov_data = np.zeros((p, p))

    for _ in range(nsample):
        indices = np.random.choice(n, size=(n,), replace=True)
        y_star = y[indices]
        X_star = X[indices]

        # Z_star = np.dot(X_star.T, y_star - pi(X_star))  # X^{*T}(y^*-X^{*T}_E\bar{\beta}_E)
        Z_star = np.dot(X_star.T, y_star - np.dot(X_star[:, active], beta_unpenalized))

        mat_XEstar = np.linalg.inv(np.dot(X_star[:, active].T, X_star[:, active]))  # (X^{*T}_E X^*_E)^{-1}
        mat_star = np.dot(np.dot(X_star[:, inactive].T, X_star[:, active]), mat_XEstar)
        data_star = np.zeros(p)
        data_star[nactive:] = Z_star[inactive,] - np.dot(mat_star, Z_star[active,])
        data_star[:nactive] = np.dot(mat_XEstar, Z_star[active,])

        _mean_cum_data += data_star
        _cov_data += np.multiply.outer(data_star, data_star)

    _cov_data /= nsample
    _mean_cum_data = _mean_cum_data / nsample
    _cov_data -= np.multiply.outer(_mean_cum_data, _mean_cum_data)

    return _cov_data
