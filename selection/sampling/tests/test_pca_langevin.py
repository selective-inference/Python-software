import numpy as np

from selection.sampling.langevin import projected_langevin

### Some examples: PCA from https://arxiv.org/abs/1410.8260
 
def _log_vandermonde(eigenvals, power=1):
    """
    Log of the Van der Monde determinant.
    """
    eigenvals = np.asarray(eigenvals)
    p = eigenvals.shape[0]
    idx = np.arange(p)
    logdiff = np.log(np.fabs(np.subtract.outer(eigenvals, eigenvals)))
    mask = np.greater.outer(idx, idx)

    return power * (logdiff * mask).sum()

def _grad_log_vandermonde(eigenvals, power=1):
    """
    Log of the Van der Monde determinant.
    """
    eigenvals = np.asarray(eigenvals)
    p = eigenvals.shape[0]
    idx = np.arange(p)
    diff = np.subtract.outer(eigenvals, eigenvals)
    diff_sign = -np.sign(diff)
    mask = (diff > 0)
    return (1. / (np.fabs(diff) + np.identity(p)) * mask * diff_sign).sum(1)

def _log_wishart_white(eigenvals, n):
    """
    Log-eigenvalue density of Wishart($I_{p \times p}$, n) assuming n>p,
    up to normalizing constant.
    """
    eigenvals = np.asarray(eigenvals)
    p = eigenvals.shape[0]

    return ((n - p - 1) * 0.5 * np.log(eigenvals).sum() 
            + _log_vandermonde(eigenvals, power=1) 
            - eigenvals.sum() * 0.5)

def _grad_log_wishart_white(eigenvals, n):
    """
    Gradient of log-eigenvalue density of Wishart($I_{p \times p}$, n) 
    assuming n>p.
    """
    eigenvals = np.asarray(eigenvals)
    p = eigenvals.shape[0]
    return ((n - p - 1) * 0.5 / (eigenvals + 1.e-7)
            + _grad_log_vandermonde(eigenvals, power=1) - 0.5)

def main(n=50):

    from sklearn.isotonic import IsotonicRegression
    import matplotlib.pyplot as plt
    initial = np.ones(n) + 0.01 * np.random.standard_normal(n)
    grad_map = lambda val: _grad_log_wishart_white(val, n)

    def projection_map(vals):
        iso = IsotonicRegression(y_min=1.e-6)
        vals = np.asarray(vals)
        return np.maximum(vals, 1.e-6)

    sampler = projected_langevin(initial,
                                 grad_map,
                                 projection_map,
                                 0.01)
    sampler = iter(sampler)

    path = [initial.copy()]
    for _ in range(200):
        print(sampler.state)
        sampler.next()
        path.append(sampler.state.copy())
    path = np.array(path)

    [plt.plot(path[:,i]) for i in range(5)]
    plt.show()

