import numpy as np, os, itertools
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr

from .comparison_metrics import (sim_xy,
                                 glmnet_lasso,
                                 relative_risk)
from .risk_comparisons import risk_comparison

def output_file(n=200, 
                p=500, 
                rho=0.35, 
                s=5, 
                beta_type=1, 
                snr_values=np.array([0.10, 0.15, 0.20, 0.25, 0.30,
                                     0.35, 0.42, 0.71, 1.22, 2.07]),
                tuning_nonrand="lambda.1se", 
                tuning_rand="lambda.1se",
                randomizing_scale=np.sqrt(0.50), 
                ndraw=50, 
                outpath = None):

    df_risk = pd.DataFrame()
    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    for snr in snr_values:
        snr_list.append(snr)
        relative_risk = np.squeeze(risk_comparison(n=n, 
                                                   p=p, 
                                                   nval=n, 
                                                   rho=rho, 
                                                   s=s, 
                                                   beta_type=beta_type, 
                                                   snr=snr,
                                                   randomizer_scale=randomizing_scale, 
                                                   full_dispersion=full_dispersion,
                                                   tuning_nonrand =tuning_nonrand, 
                                                   tuning_rand=tuning_rand, ndraw = ndraw))

        df_risk = df_risk.append(pd.DataFrame(data=relative_risk.reshape((1, 6)), columns=['sel-MLE', 'ind-est', 'rand-LASSO',
                                                                            'rel-rand-LASSO', 'rel-LASSO','LASSO']), ignore_index=True)

    df_risk['n'] = n
    df_risk['p'] = p
    df_risk['s'] = s
    df_risk['rho'] = rho
    df_risk['beta-type'] = beta_type
    df_risk['snr'] = pd.Series(np.asarray(snr_list))
    df_risk['target'] = "selected"

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_risk_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + "_rho_" + str(rho) + ".csv")
    outfile_risk_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + "_rho_" + str(rho) + ".html")
    df_risk.to_csv(outfile_risk_csv, index=False)
    df_risk.to_html(outfile_risk_html)

