import functools, hashlib

import numpy as np
from scipy.stats import norm as normal_dbn

import regreg.api as rr

from selection.algorithms.debiased_lasso import pseudoinverse_debiasing_matrix

# load in the X matrix

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.r('library(hdi); data(riboflavin); X = riboflavin$x')
numpy2ri.activate()
X_full = np.asarray(rpy.r('X')) 
numpy2ri.deactivate()

from selection.learning.utils import full_model_inference, liu_inference, pivot_plot
from selection.learning.core import split_sampler, keras_fit, repeat_selection, infer_set_target
from selection.learning.Rutils import lasso_glmnet, cv_glmnet_lam
from selection.learning.learners import mixture_learner

def highdim_model_inference(X, 
                            y,
                            truth,
                            selection_algorithm,
                            sampler,
                            lam_min,
                            dispersion,
                            success_params=(1, 1),
                            fit_probability=keras_fit,
                            fit_args={'epochs':10, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'},
                            alpha=0.1,
                            B=2000,
                            naive=True,
                            learner_klass=mixture_learner,
                            how_many=None):
    
    n, p = X.shape
    XTX = X.T.dot(X)
                         
    instance_hash = hashlib.md5()
    instance_hash.update(X.tobytes())
    instance_hash.update(y.tobytes())
    instance_hash.update(truth.tobytes())
    instance_id = instance_hash.hexdigest()

    # run selection algorithm

    observed_set = repeat_selection(selection_algorithm, sampler, *success_params)
    observed_list = sorted(observed_set)

    # observed debiased LASSO estimate

    loss = rr.squared_error(X, y)
    pen = rr.l1norm(p, lagrange=lam_min)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve()
    grad = X.T.dot(X.dot(soln) - y) # gradient at beta_hat

    M = pseudoinverse_debiasing_matrix(X,
                                       observed_list)

    observed_target = soln[observed_list] - M.dot(grad)
    tmp = X.dot(M.T)
    target_cov = tmp.T.dot(tmp) * dispersion
    cross_cov = np.identity(p)[:,observed_list] * dispersion

    if len(observed_list) > 0:

        if how_many is None:
            how_many = len(observed_list)
        observed_list = observed_list[:how_many]

        # find the target, based on the observed outcome

        (pivots, 
         covered, 
         lengths, 
         pvalues,
         lower,
         upper) = [], [], [], [], [], []

        targets = []
        true_target = truth[observed_list]

        results = infer_set_target(selection_algorithm,
                                   observed_set,
                                   observed_list,
                                   sampler,
                                   observed_target,
                                   target_cov,
                                   cross_cov,
                                   hypothesis=true_target,
                                   fit_probability=fit_probability,
                                   fit_args=fit_args,
                                   success_params=success_params,
                                   alpha=alpha,
                                   B=B,
                                   learner_klass=learner_klass)

        for i, result in enumerate(results):

            (pivot, 
             interval,
             pvalue,
             _) = result

            pvalues.append(pvalue)
            pivots.append(pivot)
            covered.append((interval[0] < true_target[i]) * (interval[1] > true_target[i]))
            lengths.append(interval[1] - interval[0])
            lower.append(interval[0])
            upper.append(interval[1])

        if len(pvalues) > 0:
            df = pd.DataFrame({'pivot':pivots,
                               'pvalue':pvalues,
                               'coverage':covered,
                               'length':lengths,
                               'upper':upper,
                               'lower':lower,
                               'id':[instance_id]*len(pvalues),
                               'target':true_target,
                               'variable':observed_list,
                               'B':[B]*len(pvalues)})
            if naive:

                (naive_pvalues, 
                 naive_pivots, 
                 naive_covered, 
                 naive_lengths, 
                 naive_upper, 
                 naive_lower) =  [], [], [], [], [], []

                for j, idx in enumerate(observed_list):
                    true_target = truth[idx]
                    target_sd = np.sqrt(target_cov[j, j])
                    observed_target_j = observed_target[j]
                    quantile = normal_dbn.ppf(1 - 0.5 * alpha)
                    naive_interval = (observed_target_j - quantile * target_sd, 
                                      observed_target_j + quantile * target_sd)
                    naive_upper.append(naive_interval[1])
                    naive_lower.append(naive_interval[0])
                    naive_pivot = (1 - normal_dbn.cdf((observed_target_j - true_target) / target_sd))
                    naive_pivot = 2 * min(naive_pivot, 1 - naive_pivot)
                    naive_pivots.append(naive_pivot)

                    naive_pvalue = (1 - normal_dbn.cdf(observed_target_j / target_sd))
                    naive_pvalue = 2 * min(naive_pvalue, 1 - naive_pvalue)
                    naive_pvalues.append(naive_pvalue)

                    naive_covered.append((naive_interval[0] < true_target) * (naive_interval[1] > true_target))
                    naive_lengths.append(naive_interval[1] - naive_interval[0])

                naive_df = pd.DataFrame({'naive_pivot':naive_pivots,
                                         'naive_pvalue':naive_pvalues,
                                         'naive_coverage':naive_covered,
                                         'naive_length':naive_lengths,
                                         'naive_upper':naive_upper,
                                         'naive_lower':naive_lower,
                                         'variable':observed_list,
                                         })

                df = pd.merge(df, naive_df, on='variable')
            return df

boot_design = False

def simulate(s=10, signal=(0.5, 1), sigma=2, alpha=0.1, B=3000, seed=0):

    # description of statistical problem

    n, p = X_full.shape

    if boot_design:
        idx = np.random.choice(np.arange(n), n, replace=True)
        X = X_full[idx] # bootstrap X to make it really an IID sample, i.e. don't condition on X throughout
        X += 0.1 * np.std(X) * np.random.standard_normal(X.shape) # to make non-degenerate
    else:
        X = X_full.copy()

    X = X - np.mean(X, 0)[None, :]
    X = X / np.std(X, 0)[None, :]

    truth = np.zeros(p)
    truth[:s] = np.linspace(signal[0], signal[1], s)
    np.random.shuffle(truth)
    truth *= sigma / np.sqrt(n)

    y = X.dot(truth) + sigma * np.random.standard_normal(n)

    lam_min, lam_1se = cv_glmnet_lam(X.copy(), y.copy(), seed=seed)
    lam_min, lam_1se = n * lam_min, n * lam_1se

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = sigma**2
                         
    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(X, XTXi, resid, sampler):

        S = sampler.center.copy()
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        G = lasso_glmnet(X, ynew, *[None]*4)
        select = G.select(seed=seed)
        return set(list(select[0]))

    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

    # run selection algorithm

    df = highdim_model_inference(X,
                                 y,
                                 truth,
                                 selection_algorithm,
                                 splitting_sampler,
                                 lam_min,
                                 sigma**2, # dispersion assumed known for now
                                 success_params=(1, 1),
                                 B=B,
                                 fit_probability=keras_fit,
                                 fit_args={'epochs':10, 'sizes':[100]*5, 'dropout':0., 'activation':'relu'})

    if df is not None:
        liu_df = liu_inference(X,
                               y,
                               1.00001 * lam_min,
                               dispersion,
                               truth,
                               alpha=alpha,
                               approximate_inverse='BN')

        return pd.merge(df, liu_df, on='variable')
    
if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    U = np.linspace(0, 1, 101)
    plt.clf()

    init_seed = np.fabs(np.random.standard_normal() * 500)
    for i in range(500):
        df = simulate(seed=init_seed+i)
        csvfile = 'riboflavin_CV.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try:
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, lengths_ax = pivot_plot(df, outbase)
                liu_pivot = df['liu_pivot']
                liu_pivot = liu_pivot[~np.isnan(liu_pivot)]
                pivot_ax.plot(U, sm.distributions.ECDF(liu_pivot)(U), 'gray', label='Liu CV',
                              linewidth=3)
                pivot_ax.legend()
                fig = pivot_ax.figure
                fig.savefig(csvfile[:-4] + '.pdf')

