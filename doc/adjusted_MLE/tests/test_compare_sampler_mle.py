import numpy as np, os, itertools
import pandas as pd

from .comparison_metrics import (sim_xy,
                                 selInf_R,
                                 glmnet_lasso,
                                 coverage,
                                 compare_sampler_MLE)

def compare_sampler_mle(n=500, 
                        p=100, 
                        rho=0.35, 
                        s=5, 
                        beta_type=1, 
                        snr_values=np.array([0.10, 0.15, 0.20, 0.25, 0.30,
                                             0.35, 0.42, 0.71, 1.22, 2.07]),
                        target="selected", 
                        tuning_rand="lambda.1se", 
                        randomizing_scale= np.sqrt(0.50), 
                        ndraw=50, 
                        outpath=None):
    
    df_selective_inference = pd.DataFrame()

    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    for snr in snr_values:
        snr_list.append(snr*np.ones(2))
        output_overall = np.zeros(23)
        for i in range(ndraw):
            output_overall += np.squeeze(
                compare_sampler_MLE(n=n, 
                                    p=p, 
                                    nval=n, 
                                    rho=rho, 
                                    s=s, 
                                    beta_type=beta_type, 
                                    snr=snr, 
                                    target = target,
                                    randomizer_scale=randomizing_scale, 
                                    full_dispersion=full_dispersion, 
                                    tuning_rand=tuning_rand))

        nreport = output_overall[22]
        randomized_MLE_inf = np.hstack(((output_overall[0:7] / 
                                         float(ndraw - nreport)).reshape((1, 7)),
                                       (output_overall[7:11] /
                                        float(ndraw)).reshape((1, 4))))
        randomized_sampler_inf = np.hstack(((output_overall[11:18] / 
                                             float(ndraw - nreport)).reshape((1, 7)),
                                        (output_overall[18:22] / 
                                         float(ndraw)).reshape((1, 4))))

        df_MLE = pd.DataFrame(data=randomized_MLE_inf, columns=['coverage', 
                                                                'length', 
                                                                'prop-infty', 
                                                                'tot-active',
                                                                'bias', 
                                                                'sel-power', 
                                                                'time',
                                                                'power', 
                                                                'power-BH', 
                                                                'fdr-BH', 
                                                                'tot-discoveries'])

        df_MLE['method'] = "MLE"
        df_sampler = pd.DataFrame(data=randomized_sampler_inf, columns=['coverage', 
                                                                        'length', 
                                                                        'prop-infty', 
                                                                        'tot-active', 
                                                                        'bias', 
                                                                        'sel-power', 
                                                                        'time',
                                                                        'power', 
                                                                        'power-BH', 
                                                                        'fdr-BH',
                                                                        'tot-discoveries'])
        df_sampler['method'] = "Sampler"

        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_sampler, ignore_index=True)

    snr_list = list(itertools.chain.from_iterable(snr_list))
    df_selective_inference['n'] = n
    df_selective_inference['p'] = p
    df_selective_inference['s'] = s
    df_selective_inference['rho'] = rho
    df_selective_inference['beta-type'] = beta_type
    df_selective_inference['snr'] = pd.Series(np.asarray(snr_list))
    df_selective_inference['target'] = target

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = (os.path.join(outpath, "compare_" + str(n) + 
                                    "_" + str(p) + "_inference_betatype" + 
                                    str(beta_type) + target + "_rho_" + str(rho) + ".csv"))
    outfile_inf_html = os.path.join(outpath, "compare_" + str(n) + 
                                    "_" + str(p) + "_inference_betatype" + 
                                    str(beta_type) + target + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)


