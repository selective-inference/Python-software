"""
We demonstrate that our optimization variables have
the correct distribution given the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

from selection.randomized.tests.test_sampling import test_conditional_law

def main(ndraw=50000, burnin=5000, remove_atom=False, unpenalized=True, stepsize=1.e-2):

    fig_idx = 0
    for (rand,
         mcmc_opt, 
         mcmc_omega,
         truncated_opt,
         truncated_omega) in test_conditional_law(ndraw=ndraw, burnin=burnin, stepsize=stepsize, unpenalized=unpenalized):

        fig_idx += 1
        fig = plt.figure(num=fig_idx, figsize=(8,8))

        plt.clf()
        idx = 0
        for i in range(mcmc_opt.shape[1]):
            plt.subplot(3,3,idx+1)

            mcmc_ = mcmc_opt[:, i]
            truncated_ = truncated_opt[:, i]

            xval = np.linspace(min(mcmc_.min(), truncated_.min()), 
                               max(mcmc_.max(), truncated_.max()), 
                               200)

            if remove_atom:
                mcmc_ = mcmc_[mcmc_ < np.max(mcmc_)]
                mcmc_ = mcmc_[mcmc_ > np.min(mcmc_)]

            plt.plot(xval, ECDF(mcmc_)(xval), label='MCMC')
            plt.plot(xval, ECDF(truncated_)(xval), label='truncated')
            idx += 1
            if idx == 1:
                plt.legend(loc='lower right')

        fig.suptitle(' '.join([rand, "opt"]))

        fig_idx += 1
        fig = plt.figure(num=fig_idx, figsize=(8,8))
        plt.clf()
        idx = 0
        for i in range(mcmc_opt.shape[1]):
            plt.subplot(3,3,idx+1)

            mcmc_ = mcmc_omega[:, i]
            truncated_ = truncated_omega[:, i]

            xval = np.linspace(min(mcmc_.min(), truncated_.min()), 
                               max(mcmc_.max(), truncated_.max()), 
                               200)

            if remove_atom:
                mcmc_ = mcmc_[mcmc_ < np.max(mcmc_)]
                mcmc_ = mcmc_[mcmc_ > np.min(mcmc_)]
            plt.plot(xval, ECDF(mcmc_)(xval), label='MCMC')
            plt.plot(xval, ECDF(truncated_)(xval), label='truncated')
            idx += 1
            if idx == 1:
                plt.legend(loc='lower right')

        fig.suptitle(' '.join([rand, "omega"]))
    plt.show()

            
            
    

