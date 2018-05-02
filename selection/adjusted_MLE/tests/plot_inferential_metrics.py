import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle

df = pd.read_csv('/Users/snigdhapanigrahi/adjusted_MLE/results/metrics_selected_target_medium.csv')
df_risk = pd.read_csv('/Users/snigdhapanigrahi/adjusted_MLE/results/risk_selected_target_medium.csv')
order = ["Selective", "Lee", "Naive"]
cols = ["#3498db", "#9b59b6", "#e74c3c"]

def inference_result():
    # Create a figure for comparing risk, coverage, lengths and power
    sns.set(font_scale=2)  # font size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                            })

    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    sns.pointplot(x="SNR", y="coverage", hue_order=order, markers='o', hue="method", data=df, ax=ax1,
                  palette=cols)
    sns.pointplot(x="SNR", y="power", hue_order=order, markers='o', hue="method", data=df, ax=ax2,
                  palette=cols)
    sns.pointplot(x="SNR", y="risk", hue_order=order, markers='o', hue="method", data=df, ax=ax3,
                  palette=cols)

    ax1.set_title("coverage", y=1.01)
    ax2.set_title("power", y=1.01)
    ax3.set_title("risk", y=1.01)

    ax1.legend_.remove()
    ax2.legend_.remove()
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax3.set_ylim(-0.05, 0.8)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    # myLocator = mticker.MultipleLocator(2)
    # ax1.xaxis.set_major_locator(myLocator)
    # ax2.xaxis.set_major_locator(myLocator)
    # ax3.xaxis.set_major_locator(myLocator)

    def common_format(ax):
        ax.grid(True, which='both')
        ax.set_xlabel('', fontsize=22)
        # ax.yaxis.label.set_size(22)
        ax.set_ylabel('', fontsize=22)
        return ax

    common_format(ax1)
    common_format(ax2)
    common_format(ax3)
    fig.text(0.5, -0.04, 'SNR', fontsize=22, ha='center')

    # add target coverage on the first plot
    ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('/Users/snigdhapanigrahi/adjusted_MLE/results/inference_comparison_medium.pdf', format='pdf', bbox_inches='tight')

def risk_comparison():
    # Create a figure for comparing risk, coverage, lengths and power
    sns.set(font_scale=2)  # font size
    sns.set_style("white", {'axes.facecolor': 'white',
                            'axes.grid': True,
                            'axes.linewidth': 2.0,
                            'grid.linestyle': u'--',
                            'grid.linewidth': 4.0,
                            'xtick.major.size': 5.0,
                            })

    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    sns.pointplot(x="SNR", y="Risk_selMLE", markers='o', data=df_risk, ax=ax1, color="#3498db")
    sns.pointplot(x="SNR", y="Risk_indest", hue_order=order, markers='o', data=df_risk, ax=ax1, color="#3498db")
    sns.pointplot(x="SNR", y="Risk_LASSO_rand", hue_order=order, markers='o', data=df_risk, ax=ax1, color="#3498db")

    ax1.set_title("risk", y=1.01)

    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    #ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    # myLocator = mticker.MultipleLocator(2)
    # ax1.xaxis.set_major_locator(myLocator)
    # ax2.xaxis.set_major_locator(myLocator)
    # ax3.xaxis.set_major_locator(myLocator)

    def common_format(ax):
        ax.grid(True, which='both')
        ax.set_xlabel('', fontsize=22)
        # ax.yaxis.label.set_size(22)
        ax.set_ylabel('', fontsize=22)
        return ax

    common_format(ax1)
    common_format(ax2)
    #common_format(ax3)
    fig.text(0.5, -0.04, 'SNR', fontsize=22, ha='center')

    # add target coverage on the first plot
    ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('/Users/snigdhapanigrahi/adjusted_MLE/results/risk_comparison_medium.pdf', format='pdf', bbox_inches='tight')

risk_comparison()
#inference_result()