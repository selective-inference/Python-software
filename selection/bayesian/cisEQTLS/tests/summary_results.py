import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/Results_cisEQTLS/high_dim_test'

allFiles = glob.glob(path + "/*.txt")

list_ = []
for file_ in allFiles:
    df = pandas.read_table(file_, header = None)
    list_.append(df.as_matrix())

coverage_ad = 0.
coverage_unad = 0.
FDR = 0.
power = 0.

for i in range(len(list_)):

    mat_0 = list_[i]

    #getting the active indices and the number of active variables:
    true_signal = np.array(mat_0[:,0])
    s = int(true_signal.sum()/true_signal[0])
    active_ind = np.array(mat_0[:,1], np.bool)
    nactive = active_ind.sum()

    #coverage of adjusted and unadjusted intervals:
    coverage_ad += np.array(mat_0[:,2]).sum()/nactive
    coverage_unad += np.array(mat_0[:, 3]).sum() /nactive

    discoveries = np.array(mat_0[:,4], np.bool)
    false_discoveries = discoveries[s:].sum()
    true_discoveries = discoveries[:s].sum()

    FDR += false_discoveries/float(s)
    power += true_discoveries/float(s)



print(coverage_ad, coverage_unad, FDR, power)





