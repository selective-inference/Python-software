import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/Results_cisEQTLS/high_dim_test_5'

allFiles = glob.glob(path + "/*.txt")

list_ = []
for file_ in allFiles:
    df = pandas.read_table(file_, header = None)
    list_.append(df.as_matrix())



#also have to access X to compute the projected truth
def summary(X, list_):

    coverage_ad = 0.
    coverage_unad = 0.
    FDR = 0.
    power = 0.

    for i in range(len(list_)):

        mat_0 = list_[i]

        true_signal = np.array(mat_0[:,0])

        if true_signal[0] >0:

            s = int(true_signal.sum()/true_signal[0])
            snr = true_signal[0]

            discoveries = np.array(mat_0[:, 2], np.bool)
            false_discoveries = discoveries[s:].sum()
            true_discoveries = discoveries[:s].sum()

        else:
            s = 0
            snr = 0.
            true_discoveries = 0

        active_ind = np.array(mat_0[:, 1], np.bool)
        nactive = active_ind.sum()

        FDR += false_discoveries / max(float(discoveries.sum()), 1.)
        power += true_discoveries / float(s)

        #coverage of adjusted and unadjusted intervals:
        projection_active = X[:, active_ind].dot(np.linalg.inv(X[:, active_ind].T.dot(X[:, active_ind])))
        true_val = projection_active.T.dot(X.dot(true_signal))

        coverage_ad = np.zeros(true_val.shape[0])
        coverage_unad = np.zeros(true_val.shape[0])

        adjusted_intervals = np.hstack[(mat_0[:,3])[active_ind], (mat_0[:,4])[active_ind]]
        unadjusted_intervals = np.hstack[(mat_0[:, 5])[active_ind], (mat_0[:, 6])[active_ind]]

        for l in range(nactive):
            if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                coverage_ad[active_ind[l]] += 1
            if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                coverage_unad[active_ind[l]] += 1

        adjusted_coverage = float(coverage_ad.sum()/nactive)
        unadjusted_coverage = float(coverage_unad.sum() /nactive)

    return adjusted_coverage, unadjusted_coverage , FDR , power
