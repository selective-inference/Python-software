import numpy as np, os, itertools
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr

from .comparison_metrics import (sim_xy,
                                 selInf_R,
                                 glmnet_lasso,
                                 BHfilter,
                                 coverage,
                                 relative_risk,
                                 comparison_cvmetrics_selected,
                                 comparison_cvmetrics_full,
                                 comparison_cvmetrics_debiased)

def plotRisk(df_risk):
    robjects.r("""
               library("ggplot2")
               library("magrittr")
               library("tidyr")
               library("dplyr")
               
               plot_risk <- function(df_risk, outpath="/Users/psnigdha/adjusted_MLE/plots/", resolution=300, height= 7.5, width=15)
                { 
                   date = 1:length(unique(df_risk$snr))
                   df_risk = filter(df_risk, metric == "Full")
                   df = cbind(df_risk, date)
                   risk = df %>%
                   gather(key, value, sel.MLE, rand.LASSO, LASSO) %>%
                   ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                   geom_point(size=3) +
                   geom_line(aes(linetype=key), size=1) +
                   ylim(0.01,1.2)+
                   labs(y="relative risk", x = "Signal regimes: snr") +
                   scale_x_continuous(breaks=1:length(unique(df_risk$snr)), label = sapply(df_risk$snr, toString)) +
                   theme(legend.position="top", legend.title = element_blank())
                   indices = sort(c("sel.MLE", "rand.LASSO", "LASSO"), index.return= TRUE)$ix
                   names = c("sel-MLE", "rand-LASSO", "LASSO")
                   risk = risk + scale_color_manual(labels = names[indices], values=c("#008B8B", "#104E8B","#B22222")[indices]) +
                   scale_shape_manual(labels = names[indices], values=c(15, 17, 16)[indices]) +
                                      scale_linetype_manual(labels = names[indices], values = c(1,1,2)[indices])
                                      outfile = paste(outpath, 'risk.png', sep="")
                   ggsave(outfile, plot = risk, dpi=resolution, dev='png', height=height, width=width, units="cm")}
                """)

    robjects.pandas2ri.activate()
    r_df_risk = robjects.conversion.py2ri(df_risk)
    R_plot = robjects.globalenv['plot_risk']
    R_plot(r_df_risk)

def plotCoveragePower(df_inference):
    robjects.r("""
               library("ggplot2")
               library("magrittr")
               library("tidyr")
               library("reshape")
               library("cowplot")
               library("dplyr")
               
               plot_coverage_lengths <- function(df_inference, outpath="/Users/psnigdha/adjusted_MLE/plots/", 
                                                 resolution=200, height_plot1= 6.5, width_plot1=12, 
                                                 height_plot2=13, width_plot2=13)
               {
                 snr.len = length(unique(df_inference$snr))
                 df_inference = arrange(df_inference, method)
                 target = toString(df_inference$target[1])
                 df = data.frame(snr = sapply(unique(df_inference$snr), toString),
                                 MLE = 100*df_inference$coverage[((2*snr.len)+1):(3*snr.len)],
                                 Lee = 100*df_inference$coverage[1:snr.len],
                                 Naive = 100*df_inference$coverage[((3*snr.len)+1):(4*snr.len)])
                 if(target== "selected"){
                      data.m <- melt(df, id.vars='snr')
                      coverage = ggplot(data.m, aes(snr, value)) + 
                                 geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                                 geom_hline(yintercept = 90, linetype="dotted") +
                                 labs(y="coverage: partial", x = "Signal regimes: snr") +
                                 theme(legend.position="top", 
                                       legend.title = element_blank()) 
                      coverage = coverage + 
                                 scale_fill_manual(labels = c("MLE-based","Lee", "Naive"), values=c("#008B8B", "#B22222", "#FF6347"))} else{
                 df = cbind(df, Liu = 100*df_inference$coverage[((snr.len)+1):(2*snr.len)])
                 df <- df[c("snr", "MLE", "Liu", "Lee", "Naive")]
                 data.m <- melt(df, id.vars='snr')
                 coverage = ggplot(data.m, aes(snr, value)) + 
                            geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                            geom_hline(yintercept = 90, linetype="dotted") +
                            labs(y="coverage: full", x = "Signal regimes: snr") +
                            theme(legend.position="top", legend.title = element_blank()) 
                  coverage = coverage + 
                  scale_fill_manual(labels = c("MLE-based", "Liu", "Lee", "Naive"), values=c("#008B8B", "#104E8B", "#B22222", "#FF6347"))}
  
                 outfile = paste(outpath, 'coverage.png', sep="")
                 ggsave(outfile, plot = coverage, dpi=resolution, dev='png', height=height_plot1, width=width_plot1, units="cm")
               
                 df = data.frame(snr = sapply(unique(df_inference$snr), toString),
                                 MLE = 100*df_inference$sel.power[((2*snr.len)+1):(3*snr.len)],
                                 Lee = 100*df_inference$sel.power[1:snr.len])
                 if(target== "selected"){
                   data.m <- melt(df, id.vars='snr')
                   sel_power = ggplot(data.m, aes(snr, value)) + 
                               geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                               labs(y="power: partial", x = "Signal regimes: snr") +
                               theme(legend.position="top", legend.title = element_blank()) 
                   sel_power = sel_power + scale_fill_manual(labels = c("MLE-based","Lee"), values=c("#008B8B", "#B22222"))} else{
                   df = cbind(df, Liu = 100*df_inference$sel.power[((snr.len)+1):(2*snr.len)])
                   df <- df[,c("snr", "MLE", "Liu", "Lee")]
                   data.m <- melt(df, id.vars='snr')
                   sel_power = ggplot(data.m, aes(snr, value)) + 
                               geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                               labs(y="power: full", x = "Signal regimes: snr") +
                               theme(legend.position="top", legend.title = element_blank()) 
                   sel_power = sel_power + scale_fill_manual(labels = c("MLE-based","Liu","Lee"), values=c("#008B8B", "#104E8B", "#B22222"))}
  
                 outfile = paste(outpath, 'selective_power.png', sep="")
                 ggsave(outfile, plot = sel_power, dpi=resolution, dev='png', height=height_plot1, width=width_plot1, units="cm")
  
               if(target== "selected"){
                   test_data <-data.frame(MLE = filter(df_inference, method == "MLE")$length,
                   Lee = filter(df_inference, method == "Lee")$length,
                   Naive = filter(df_inference, method == "Naive")$length,
                   date = 1:length(unique(df_inference$snr)))
                   lengths = test_data %>%
                             gather(key, value, MLE, Lee, Naive) %>%
                             ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                             geom_point(size=3) +
                             geom_line(aes(linetype=key), size=1) +
                             ylim(0.,max(test_data$MLE, test_data$Lee, test_data$Naive) + 0.2)+
                             labs(y="lengths:partial", x = "Signal regimes: snr") +
                             scale_x_continuous(breaks=1:length(unique(df_inference$snr)), label = sapply(unique(df_inference$snr), toString))+
                             theme(legend.position="top", legend.title = element_blank())
    
                   indices = sort(c("MLE", "Lee", "Naive"), index.return= TRUE)$ix
                   names = c("MLE-based", "Lee", "Naive")
                   lengths = lengths + scale_color_manual(labels = names[indices], values=c("#008B8B","#B22222", "#FF6347")[indices]) +
                             scale_shape_manual(labels = names[indices], values=c(15, 17, 16)[indices]) +
                             scale_linetype_manual(labels = names[indices], values = c(1,1,2)[indices])} else{
                   test_data <-data.frame(MLE = filter(df_inference, method == "MLE")$length,
                                          Lee = filter(df_inference, method == "Lee")$length,
                                          Naive = filter(df_inference, method == "Naive")$length,
                                          Liu = filter(df_inference, method == "Liu")$length,
                                          date = 1:length(unique(df_inference$snr)))
                   lengths= test_data %>%
                            gather(key, value, MLE, Lee, Naive, Liu) %>%
                            ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                            geom_point(size=3) +
                            geom_line(aes(linetype=key), size=1) +
                            ylim(0.,max(test_data$MLE, test_data$Lee, test_data$Naive, test_data$Liu) + 0.2)+
                            labs(y="lengths: full", x = "Signal regimes: snr") +
                            scale_x_continuous(breaks=1:length(unique(df_inference$snr)), label = sapply(unique(df_inference$snr), toString))+
                            theme(legend.position="top", legend.title = element_blank())
         
                   indices = sort(c("MLE", "Liu", "Lee", "Naive"), index.return= TRUE)$ix
                   names = c("MLE-based", "Lee", "Naive", "Liu")
                   lengths = lengths + scale_color_manual(labels = names[indices], values=c("#008B8B","#B22222", "#FF6347", "#104E8B")[indices]) +
                             scale_shape_manual(labels = names[indices], values=c(15, 17, 16, 15)[indices]) +
                             scale_linetype_manual(labels = names[indices], values = c(1,1,2,1)[indices])}
  
               prop = filter(df_inference, method == "Lee")$prop.infty
               df = data.frame(snr = sapply(unique(df_inference$snr), toString),
               infinite = 100*prop)
               data.prop <- melt(df, id.vars='snr')
               pL = ggplot(data.prop, aes(snr, value)) +
                    geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                    labs(y="infinite intervals (%)", x = "Signal regimes: snr") +
                    theme(legend.position="top", 
                    legend.title = element_blank()) 
               pL = pL + scale_fill_manual(labels = c("Lee"), values=c("#B22222"))
               prow <- plot_grid( pL + theme(legend.position="none"),
                                  lengths + theme(legend.position="none"),
                                  align = 'vh',
                                  hjust = -1,
                                  ncol = 1)
  
               legend <- get_legend(lengths+ theme(legend.direction = "horizontal",legend.justification="center" ,legend.box.just = "bottom"))
               p <- plot_grid(prow, ncol=1, legend, rel_heights = c(2., .2)) 
               outfile = paste(outpath, 'length.png', sep="")
               ggsave(outfile, plot = p, dpi=resolution, dev='png', height=height_plot2, width=width_plot2, units="cm")}
               """)

    robjects.pandas2ri.activate()
    r_df_inference = robjects.conversion.py2ri(df_inference)
    R_plot = robjects.globalenv['plot_coverage_lengths']
    R_plot(r_df_inference)

def output_file(n=500, p=100, rho=0.35, s=5, beta_type=1, snr_values=np.array([0.10, 0.15, 0.20, 0.25, 0.30,
                                                                               0.35, 0.42, 0.71, 1.22, 2.07]),
                target="selected", tuning_nonrand="lambda.1se", tuning_rand="lambda.1se",
                randomizing_scale = np.sqrt(0.50), ndraw = 50, outpath = None, plot=False):

    df_selective_inference = pd.DataFrame()
    df_risk = pd.DataFrame()

    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    snr_list_0 = []
    for snr in snr_values:
        snr_list.append(snr*np.ones(4))
        snr_list_0.append(snr*np.ones(2))
        output_overall = np.zeros(55)
        if target == "selected":
            for i in range(ndraw):
                output_overall += np.squeeze(comparison_cvmetrics_selected(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                                           randomizer_scale=randomizing_scale, full_dispersion=full_dispersion,
                                                                           tuning_nonrand =tuning_nonrand, tuning_rand=tuning_rand))
        elif target == "full":
            for i in range(ndraw):
                output_overall += np.squeeze(comparison_cvmetrics_full(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                                       randomizer_scale=randomizing_scale, full_dispersion=full_dispersion,
                                                                       tuning_nonrand =tuning_nonrand, tuning_rand=tuning_rand))
        elif target == "debiased":
            for i in range(ndraw):
                output_overall += np.squeeze(comparison_cvmetrics_debiased(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                                           randomizer_scale=randomizing_scale, full_dispersion=full_dispersion,
                                                                           tuning_nonrand =tuning_nonrand, tuning_rand=tuning_rand))

        nLee = output_overall[52]
        nLiu = output_overall[53]
        nMLE = output_overall[54]

        relative_risk = (output_overall[0:6] / float(ndraw)).reshape((1, 6))
        partial_risk = np.hstack(((output_overall[46:50] / float(ndraw-nMLE)).reshape((1, 4)),
                                  (output_overall[50:52] / float(ndraw - nLee)).reshape((1, 2))))

        nonrandomized_naive_inf = np.hstack(((output_overall[6:12] / float(ndraw - nLee)).reshape((1, 6)),
                                             (output_overall[12:16] / float(ndraw)).reshape((1, 4))))
        nonrandomized_Lee_inf = np.hstack(((output_overall[16:22] / float(ndraw - nLee)).reshape((1, 6)),
                                          (output_overall[22:26] / float(ndraw)).reshape((1, 4))))
        nonrandomized_Liu_inf = np.hstack(((output_overall[26:32] / float(ndraw - nLiu)).reshape((1, 6)),
                                          (output_overall[32:36] / float(ndraw)).reshape((1, 4))))
        randomized_MLE_inf = np.hstack(((output_overall[36:42] / float(ndraw - nMLE)).reshape((1, 6)),
                                       (output_overall[42:46] / float(ndraw)).reshape((1, 4))))

        if target=="selected":
            nonrandomized_Liu_inf[nonrandomized_Liu_inf==0] = 'NaN'
        if target == "debiased":
            nonrandomized_Liu_inf[nonrandomized_Liu_inf == 0] = 'NaN'
            nonrandomized_Lee_inf[nonrandomized_Lee_inf == 0] = 'NaN'

        df_naive = pd.DataFrame(data=nonrandomized_naive_inf,columns=['coverage', 'length', 'prop-infty', 'tot-active', 'bias', 'sel-power',
                                                                      'power', 'power-BH', 'fdr-BH','tot-discoveries'])
        df_naive['method'] = "Naive"
        df_Lee = pd.DataFrame(data=nonrandomized_Lee_inf, columns=['coverage', 'length', 'prop-infty','tot-active','bias', 'sel-power',
                                                                   'power', 'power-BH', 'fdr-BH','tot-discoveries'])
        df_Lee['method'] = "Lee"

        df_Liu = pd.DataFrame(data=nonrandomized_Liu_inf,columns=['coverage', 'length', 'prop-infty', 'tot-active','bias', 'sel-power',
                                                                  'power', 'power-BH', 'fdr-BH', 'tot-discoveries'])
        df_Liu['method'] = "Liu"

        df_MLE = pd.DataFrame(data=randomized_MLE_inf, columns=['coverage', 'length', 'prop-infty', 'tot-active','bias', 'sel-power',
                                                                'power', 'power-BH', 'fdr-BH', 'tot-discoveries'])
        df_MLE['method'] = "MLE"

        df_risk_metrics = pd.DataFrame(data=relative_risk, columns=['sel-MLE', 'ind-est', 'rand-LASSO','rel-rand-LASSO', 'rel-LASSO', 'LASSO'])
        df_risk_metrics['metric'] = "Full"
        df_prisk_metrics = pd.DataFrame(data=partial_risk,columns=['sel-MLE', 'ind-est', 'rand-LASSO', 'rel-rand-LASSO', 'rel-LASSO','LASSO'])
        df_prisk_metrics['metric'] = "Partial"

        df_selective_inference = df_selective_inference.append(df_naive, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_Lee, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_Liu, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)

        df_risk = df_risk.append(df_risk_metrics, ignore_index=True)
        df_risk = df_risk.append(df_prisk_metrics, ignore_index=True)

    snr_list = list(itertools.chain.from_iterable(snr_list))
    df_selective_inference['n'] = n
    df_selective_inference['p'] = p
    df_selective_inference['s'] = s
    df_selective_inference['rho'] = rho
    df_selective_inference['beta-type'] = beta_type
    df_selective_inference['snr'] = pd.Series(np.asarray(snr_list))
    df_selective_inference['target'] = target

    snr_list_0 = list(itertools.chain.from_iterable(snr_list_0))
    df_risk['n'] = n
    df_risk['p'] = p
    df_risk['s'] = s
    df_risk['rho'] = rho
    df_risk['beta-type'] = beta_type
    df_risk['snr'] = pd.Series(np.asarray(snr_list_0))
    df_risk['target'] = target

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_risk_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    outfile_risk_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_risk.to_csv(outfile_risk_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)
    df_risk.to_html(outfile_risk_html)

    if plot is True:
        plotRisk(df_risk)
        plotCoveragePower(df_selective_inference)






















