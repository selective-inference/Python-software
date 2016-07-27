import numpy as np
from scipy.stats import laplace, probplot, uniform

from matplotlib import pyplot as plt
from selection.sampling.langevin import projected_langevin
from selection.distributions.discrete_family import discrete_family


def test_simple_problem(n=100, randomization_dist = "logistic", threshold =1,
                        weights="normal",
                        Langevin_steps=6000, burning = 1000):
    step_size = 1./n

    y = np.random.standard_normal(n)
    obs = np.sqrt(n)*np.mean(y)

    if randomization_dist=="logistic":
        omega = np.random.logistic(loc=0, scale=1, size=1)

    if (obs+omega<threshold):
        return -1

    initial_state = np.ones(n)
    #initial_state = np.zeros(n)

    centered_y = y-np.mean(y)

    def full_projection(state):
        return state

    def full_gradient(state, n=n):

        gradient = np.zeros(n)
        if weights == "normal":
            gradient -= state
        if (weights == "gumbel"):
            gumbel_beta = np.sqrt(6) / (1.14 * np.pi)
            euler = 0.57721
            gumbel_mu = -gumbel_beta * euler
            gumbel_sigma = 1. / 1.14
            gradient -= (1. - np.exp(-(state * gumbel_sigma - gumbel_mu) / gumbel_beta)) * gumbel_sigma / gumbel_beta
        if weights == "logistic":
            gradient = np.divide(np.exp(-state)-1, np.exp(-state)+1)

        y_cs = centered_y / np.sqrt(n)
        if weights =="neutral":
            gradient = - np.inner(state, y_cs) * y_cs

        omega = -np.inner(y_cs, state) + threshold
        if randomization_dist=="logistic":
            randomization_derivative = -1./(1+np.exp(-omega))

        gradient -= y_cs * randomization_derivative

        return gradient


    sampler = projected_langevin(initial_state.copy(),
                                 full_gradient,
                                 full_projection,
                                 step_size)

    samples = []

    for i in range(Langevin_steps):
        sampler.next()
        if (i > burning):
            samples.append(sampler.state.copy())

    alphas = np.array(samples)

    pop = [np.inner(centered_y, alphas[i,:])/np.sqrt(n) for i in range(alphas.shape[0])]

    pop = np.abs(pop)

    fam = discrete_family(pop, np.ones_like(pop))
    pval = fam.cdf(0, np.abs(obs))
    pval = 2 * min(pval, 1 - pval)
    print "observed: ", obs, "p value: ", pval
    return pval


if __name__ == "__main__":

    np.random.seed(1)
    plt.figure()
    plt.ion()
    P0 = []
    for i in range(500):
        print "iteration", i
        p0 = test_simple_problem()
        if p0>-1:
            P0.append(p0)
        plt.clf()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        probplot(P0, dist=uniform, sparams=(0, 1), plot=plt,fit=False)
        plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
        plt.pause(0.01)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)

    while True:
        plt.pause(0.05)

