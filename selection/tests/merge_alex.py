import numpy as np
import os, glob

fs = glob.glob('*alex*npy')
simpler = set([f.replace('alex1', 'alex').replace('alex0', 'alex').replace('alexnull0', 'alexnull').replace('alexnull1', 'alexnull') for f in fs])

for s in simpler:
    if 'null' in s:
        data = np.vstack([np.load(s.replace('alexnull', 'alexnull1')), np.load(s.replace('alexnull', 'alexnull0'))])
    else:
        data = np.vstack([np.load(s.replace('alex', 'alex1')), np.load(s.replace('alex', 'alex0'))])
    np.save(os.path.join('..', s), data)
