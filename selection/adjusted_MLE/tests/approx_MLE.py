import numpy as np
from scipy.stats import norm as ndist
from scipy.optimize import minimize

def log_barrier(u, barrier_scale, threshold = 2.):

    BIG = 10 ** 10
    violation = u-threshold<0.
    return np.log(1 + (np.sqrt(barrier_scale)/ (u-threshold))) + violation* BIG

def grad_log_barrier(u, barrier_scale, threshold = 2.):
    return 1./(u-threshold + np.sqrt(barrier_scale)) - 1./(u-threshold)

def grad_log_hessian(u, barrier_scale, threshold = 2.):
    return -1. / ((u - threshold + np.sqrt(barrier_scale))**2.) + 1. / ((u - threshold)** 2.)

def approx_grad_cgf(mu, randomization_scale = 0.5, threshold = 2, nstep= 50, tol=1.e-10):

    variance = 1 + randomization_scale ** 2.
    objective = lambda u: -u*(mu/variance) + (u ** 2.)/(2.* variance)+ log_barrier(u, variance)
    gradient = lambda u: -(mu/variance) + u/variance + grad_log_barrier(u, variance)
    hessian = lambda u: 1/variance + grad_log_hessian(u, variance)

    current_value = np.inf
    initial = threshold +1.
    current = initial
    step = 1

    for itercount in range(nstep):
        newton_step = (gradient(current)/(hessian(current)))

        # make sure proposal is feasible
        count = 0
        while True:
            count += 1
            proposal = current - step * newton_step
            failing = (proposal < threshold)
            if not failing.sum():
                break
            step *= 0.5 ** failing

            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        while True:
            proposal = current - step * newton_step
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    value = objective(current)
    return current/variance + ((randomization_scale** 2.)/(1+randomization_scale**2.))*mu, value, current

def approx_fisher_info(mu, randomization_scale=0.5, threshold=2):

    variance = 1 + randomization_scale ** 2.
    minimizer = approx_grad_cgf(mu)[2]
    return (1./ variance**2.)* (1./((1./variance) + grad_log_hessian(minimizer, variance)))+ ((randomization_scale ** 2.)/variance)

def simulate_truncated(mu, randomization_scale = 0.5, threshold = 2):
    while True:
        Z = np.random.normal(mu, 1, 1)
        W = np.random.normal(0, randomization_scale, 1)
        if (Z + W > threshold):
            return Z

def test_pivot(mu, randomization_scale=0.5, threshold=2):
    Z = np.array([simulate_truncated(mu, randomization_scale=randomization_scale, threshold=threshold) for _ in
                  range(25000)])

    mu_seq = np.linspace(-7., 6, num=2600)
    grad_partition = np.zeros(mu_seq.shape[0])
    for i in range(mu_seq.shape[0]):
        grad_partition[i] = approx_grad_cgf(mu_seq[i])[0]

    pivot = []
    approx_MLE = []
    sd_MLE = 1 / np.sqrt(approx_fisher_info(mu))
    for k in range(Z.shape[0]):
        MLE = mu_seq[np.argmin(np.abs(grad_partition - Z[k]))]
        approx_MLE.append(MLE)
        pivot.append((MLE - mu) / sd_MLE)

    return np.asarray(pivot), np.asarray(approx_MLE)

print(test_pivot(1))

    #print("grad cgf check", approx_grad_cgf(-1)[0])
#print("fisher info check", approx_fisher_info(-2))