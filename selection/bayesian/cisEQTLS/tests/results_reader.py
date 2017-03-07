
import os, numpy as np, pandas, statsmodels.api as sm

file_ = "/Users/snigdhapanigrahi/Results_cisEQTLS/infs_double_10.txt"
df = np.loadtxt(file_, delimiter=',')

print(df.shape)

len = df.shape[0]

print(df[:,0].sum()/len, df[:,1].sum()/len, float(df[:,2].sum()/len), float(df[:,3].sum()/len))