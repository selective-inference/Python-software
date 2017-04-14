import glob
import os, numpy as np, pandas, statsmodels.api as sm

#path =r'/Users/snigdhapanigrahi/Results_freq_EQTL/sparsity_5/dim_1/dim_1'
#path =r'/Users/snigdhapanigrahi/Results_bayesian/fixed_lasso/fixed_lasso'

path =r'/Users/snigdhapanigrahi/Results_bayesian/bayesian/fixed_lasso'
allFiles = glob.glob(path + "/*.txt")

list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

def summary_files(list_):

    coverage_ad = 0.
    coverage_unad = 0.
    length_ad = 0.
    length_unad = 0.

    length = len(list_)
    print("number of simulations", length)
    length = length - 50

    for i in range(length):
        print("iteration", i)
        lasso = list_[i].reshape((4, 1))
        coverage_ad += lasso[0,0]
        coverage_unad += lasso[1,0]
        length_ad += lasso[2,0]
        length_unad += lasso[3,0]

    return coverage_ad / length, coverage_unad / length, length_ad / length, length_unad / length

print(summary_files(list_))



