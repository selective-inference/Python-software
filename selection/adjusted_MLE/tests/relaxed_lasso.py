from rpy2.robjects.packages import importr
from rpy2 import robjects

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def sim_xy(n, p, nval, rho=0, s=5):
    robjects.r('''
    source('~/best-subset/bestsubset/R/sim.R')
    ''')

    r_simulate = robjects.globalenv['sim.xy']
    print(r_simulate(n, p, nval, rho=rho, s=s))

sim_xy(n=50, p=10, nval=50)