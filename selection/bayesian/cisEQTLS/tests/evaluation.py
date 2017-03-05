import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/Results_cisEQTLS/infs_single'

allFiles = glob.glob(path + "/*.txt")

list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)


def evaluation_per_file(list, X, s, snr =5.):

    FDR = 0.
    power = 0.

    n, p = X.shape
    true_beta = np.zeros(p)
    true_beta[:s] = snr

    discoveries = np.array(list[:, 1], np.bool)

    if true_beta[0] > 0:

        true_discoveries = discoveries[:s].sum()

    else:
        true_discoveries = 0

    false_discoveries = discoveries[s:].sum()
    FDR += false_discoveries / max(float(discoveries.sum()), 1.)
    power += true_discoveries / float(s)

    active_ind = np.array(list[:, 0], np.bool)
    nactive = active_ind.sum()

    projection_active = X[:, active_ind].dot(np.linalg.inv(X[:, active_ind].T.dot(X[:, active_ind])))
    true_val = projection_active.T.dot(X.dot(true_beta))

    coverage_ad = np.zeros(true_val.shape[0])
    coverage_unad = np.zeros(true_val.shape[0])

    adjusted_intervals = np.zeros((2,nactive))
    adjusted_intervals[0,:] = (list[:, 2])[active_ind]
    adjusted_intervals[1,:] = (list[:, 3])[active_ind]

    unadjusted_intervals = np.zeros((2, nactive))
    unadjusted_intervals[0, :] = (list[:, 4])[active_ind]
    unadjusted_intervals[1, :] = (list[:, 5])[active_ind]

    for l in range(nactive):
        if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
            coverage_ad[l] += 1
        if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
            coverage_unad[l] += 1

    adjusted_coverage = float(coverage_ad.sum() / nactive)
    unadjusted_coverage = float(coverage_unad.sum() / nactive)

    return adjusted_coverage, unadjusted_coverage, FDR, power


def summary_files(list_):

    coverage_ad = 0.
    coverage_unad = 0.
    FDR = 0.
    power = 0.
    length = len(list_)

    for i in range(length):
        X = np.random.standard_normal((350, 7000))
        X /= (X.std(0)[None, :] * np.sqrt(350))

        results = evaluation_per_file(list_[i], X=X, s=1, snr =5.)
        coverage_ad += results[0]
        coverage_unad += results[1]
        FDR += results[2]
        power += results[3]

    return coverage_ad/length, coverage_unad/length , FDR/length , power/length

print(summary_files(list_))