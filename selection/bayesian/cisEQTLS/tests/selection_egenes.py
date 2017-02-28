#this file shall generate the data, do the Simes selection controlling for FWER and write into a file the output
import numpy as np
import random
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from selection.tests.instance import gaussian_instance
from selection.bayesian.cisEQTLS.initial_sol_wocv import selection, instance

def Simes_selection_egenes(txtfile, ngenes = 350 , n=350, p = 5000, seed_n = 19, bh_level=0.1):

    random.seed(seed_n)

    sel_ind = np.zeros(ngenes)

    index_ssig = np.zeros(ngenes)

    indices_rej = np.zeros((ngenes,10))

    index_order = np.zeros(ngenes)

    sign_T = np.zeros(ngenes)

    snr = np.zeros(ngenes)

    s = np.zeros(ngenes)

    simes_level = 0.1 * np.ones(ngenes)

    for rep in range(ngenes):

        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=0, sigma=1, rho=0, snr=5.)

        sel_simes = simes_selection(X, y, alpha= simes_level[rep]/ ngenes, randomizer='gaussian')

        if sel_simes is not None:

            sel_ind[rep] = 1

            index_ssig[rep] = sel_simes[0]

            t_0 = sel_simes[2]

            sign_T[rep] = sel_simes[3]

            index_order[rep] = sel_simes[2]

            snr[rep] = 5.

            s[rep] = 0

            if t_0 > 0:

                (indices_rej[rep,:])[:sel_simes[1].shape[0]] = sel_simes[1]

        else:

            sel_ind[rep] = 0

    with open(txtfile, "w") as output:
        for val in range(ngenes):
            output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(sel_ind[val],
                                                       index_ssig[val],
                                                       indices_rej[val,:],
                                                       index_order[val],
                                                       sign_T[val],
                                                       snr[rep],
                                                       s[rep],
                                                       simes_level[rep]))

Simes_selection_egenes("/Users/snigdhapanigrahi/Results_cisEQTLS/output.txt", ngenes = 350 , n=350, p = 5000, seed_n = 19, bh_level=0.1)