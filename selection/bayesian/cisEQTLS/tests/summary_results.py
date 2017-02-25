import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/Results_cisEQTLS/high_dim_test'

allFiles = glob.glob(path + "/*.txt")

list_ = []
for file_ in allFiles:
    df = pandas.read_csv(file_,index_col=None, header=0)
    list_.append(df)