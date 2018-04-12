import numpy as np, sys
import pandas as pd
from rpy2 import robjects
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L;

reload(L)
from selection.randomized.lasso import highdim
from selection.algorithms.lasso import lasso
from scipy.stats import norm as ndist
from selection.adjusted_MLE.tests.test_inferential_metrics import (BHfilter,
                                                                   selInf_R,
                                                                   glmnet_lasso,
                                                                   sim_xy,
                                                                   tuned_lasso,
                                                                   relative_risk,
                                                                   coverage,
                                                                   comparison_risk_inference_selected,
                                                                   comparison_risk_inference_full)

if __name__ == "__main__":

    df_master = pd.DataFrame()
    df_risk = pd.DataFrame()

    target = "selected"
    snr_values = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.42, 0.71, 1.22])

    for snr in snr_values:
        ndraw = 50
        bias = 0.
        risk_selMLE = 0.
        risk_indest = 0.
        risk_LASSO_rand = 0.
        risk_relLASSO_rand = 0.

        risk_relLASSO_nonrand = 0.
        risk_LASSO_nonrand = 0.

        coverage_selMLE = 0.
        coverage_Lee = 0.
        coverage_unad = 0.

        length_sel = 0.
        length_Lee = 0.
        length_unad = 0.

        power_sel = 0.
        power_Lee = 0.
        power_unad = 0.
        n, p, rho, s, beta_type, snr = 500, 100, 0.35, 5, 1, snr

        if target == "selected":
            for i in range(ndraw):
                output = comparison_risk_inference_selected(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type,
                                                            snr=snr,
                                                            randomizer_scale=np.sqrt(0.25), target=target,
                                                            full_dispersion=True)

                risk_selMLE += output[0]
                risk_indest += output[1]
                risk_LASSO_rand += output[2]
                risk_relLASSO_rand += output[3]
                risk_relLASSO_nonrand += output[4]
                risk_LASSO_nonrand += output[5]

                coverage_selMLE += output[6]
                coverage_Lee += output[7]
                coverage_unad += output[8]

                length_sel += output[9]
                length_Lee += output[10]
                length_unad += output[11]

                power_sel += output[12]
                power_Lee += output[13]
                power_unad += output[14]

                sys.stderr.write("overall selMLE risk " + str(risk_selMLE / float(i + 1)) + "\n")
                sys.stderr.write("overall indep est risk " + str(risk_indest / float(i + 1)) + "\n")
                sys.stderr.write("overall randomized LASSO est risk " + str(risk_LASSO_rand / float(i + 1)) + "\n")
                sys.stderr.write(
                    "overall relaxed rand LASSO est risk " + str(risk_relLASSO_rand / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall relLASSO risk " + str(risk_relLASSO_nonrand / float(i + 1)) + "\n")
                sys.stderr.write("overall LASSO risk " + str(risk_LASSO_nonrand / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall selective coverage " + str(coverage_selMLE / float(i + 1)) + "\n")
                sys.stderr.write("overall Lee coverage " + str(coverage_Lee / float(i + 1)) + "\n")
                sys.stderr.write("overall unad coverage " + str(coverage_unad / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall selective length " + str(length_sel / float(i + 1)) + "\n")
                sys.stderr.write("overall Lee length " + str(length_Lee / float(i + 1)) + "\n")
                sys.stderr.write("overall unad length " + str(length_unad / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall selective power " + str(power_sel / float(i + 1)) + "\n")
                sys.stderr.write("overall Lee power " + str(power_Lee / float(i + 1)) + "\n")
                sys.stderr.write("overall unad power " + str(power_unad / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("iteration completed " + str(i + 1) + "\n")

                # metrics = pd.DataFrame()
                metrics_selective = pd.DataFrame({"sample_size": n,
                                                  "regression_dim": p,
                                                  "correlation": rho,
                                                  "SNR": snr,
                                                  "signal_type": beta_type,
                                                  "risk": output[0],
                                                  "coverage": output[6],
                                                  "length": output[9],
                                                  "power": output[12],
                                                  "method": "Selective"}, index=[0])

                metrics_Lee = pd.DataFrame({"sample_size": n,
                                            "regression_dim": p,
                                            "correlation": rho,
                                            "SNR": snr,
                                            "signal_type": beta_type,
                                            "risk": output[5],
                                            "coverage": output[7],
                                            "length": output[10],
                                            "power": output[13],
                                            "method": "Lee"}, index=[0])

                metrics_unad = pd.DataFrame({"sample_size": n,
                                             "regression_dim": p,
                                             "correlation": rho,
                                             "SNR": snr,
                                             "signal_type": beta_type,
                                             "risk": output[5],
                                             "coverage": output[8],
                                             "length": output[11],
                                             "power": output[14],
                                             "method": "Naive"}, index=[0])

                metrics = pd.DataFrame({"sample_size": n,
                                        "regression_dim": p,
                                        "correlation": rho,
                                        "SNR": snr,
                                        "signal_type": beta_type,
                                        "Risk_selMLE": output[0],
                                        "Risk_indest": output[1],
                                        "Risk_LASSO_rand": output[2],
                                        "Risk_relLASSO_rand": output[3],
                                        "Risk_relLASSO_nonrand": output[4],
                                        "Risk_LASSO_nonrand": output[5]}, index=[0])

                df_master = df_master.append(metrics_selective, ignore_index=True)
                df_master = df_master.append(metrics_Lee, ignore_index=True)
                df_master = df_master.append(metrics_unad, ignore_index=True)
                df_risk = df_risk.append(metrics, ignore_index=True)

        elif target == "full":
            if n > p:
                full_dispersion = True
            else:
                full_dispersion = False
            for i in range(ndraw):
                output = comparison_risk_inference_full(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                        randomizer_scale=np.sqrt(0.25), target=target,
                                                        full_dispersion=full_dispersion)

                risk_selMLE += output[0]
                risk_indest += output[1]
                risk_LASSO_rand += output[2]
                risk_relLASSO_rand += output[3]
                risk_relLASSO_nonrand += output[4]
                risk_LASSO_nonrand += output[5]

                coverage_selMLE += output[6]
                coverage_unad += output[7]

                length_sel += output[8]
                length_unad += output[9]

                power_sel += output[10]
                power_unad += output[11]

                sys.stderr.write("overall selMLE risk " + str(risk_selMLE / float(i + 1)) + "\n")
                sys.stderr.write("overall indep est risk " + str(risk_indest / float(i + 1)) + "\n")
                sys.stderr.write("overall randomized LASSO est risk " + str(risk_LASSO_rand / float(i + 1)) + "\n")
                sys.stderr.write(
                    "overall relaxed rand LASSO est risk " + str(risk_relLASSO_rand / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall relLASSO risk " + str(risk_relLASSO_nonrand / float(i + 1)) + "\n")
                sys.stderr.write("overall LASSO risk " + str(risk_LASSO_nonrand / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall selective coverage " + str(coverage_selMLE / float(i + 1)) + "\n")
                sys.stderr.write("overall unad coverage " + str(coverage_unad / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall selective length " + str(length_sel / float(i + 1)) + "\n")
                sys.stderr.write("overall unad length " + str(length_unad / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("overall selective power " + str(power_sel / float(i + 1)) + "\n")
                sys.stderr.write("overall unad power " + str(power_unad / float(i + 1)) + "\n" + "\n")

                sys.stderr.write("iteration completed " + str(i + 1) + "\n")

    df_master.to_csv("/Users/snigdhapanigrahi/adjusted_MLE/results/metrics_selected_target_medium.csv", index=False)
    df_risk.to_csv("/Users/snigdhapanigrahi/adjusted_MLE/results/risk_selected_target_medium.csv", index=False)