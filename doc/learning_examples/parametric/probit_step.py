import functools, hashlib

import numpy as np
from scipy.stats import norm as normal_dbn

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
numpy2ri.activate()

from selection.learning.learners import mixture_learner
from selection.learning.utils import naive_partial_model_inference, pivot_plot
from selection.learning.core import random_forest_fit_sk, infer_general_target

def probit_MLE(X, y, formula_terms, truth=None, alpha=0.1):

    numpy2ri.activate()
    rpy.r.assign('X', X)
    rpy.r.assign('y', y)
    rpy.r('D = data.frame(X, y)')
    rpy.r('M = glm(y ~ %s, family=binomial(link="probit"), data=D)' % 
          ' + '.join(formula_terms))
    beta_hat = rpy.r('coef(M)')
    target_cov = rpy.r('vcov(M)')

    if truth is None:
        truth = np.zeros_like(beta_hat)
    SE = np.sqrt(np.diag(target_cov))

    Z = (beta_hat - truth) / SE
    Z0 = beta_hat / SE

    pvalues = normal_dbn.cdf(Z0)
    pvalues = 2 * np.minimum(pvalues, 1 - pvalues) 

    pivots = normal_dbn.cdf(Z)
    pivots = 2 * np.minimum(pivots, 1 - pivots) 

    upper = beta_hat + normal_dbn.ppf(1 - 0.5 * alpha) * SE
    lower = beta_hat - normal_dbn.ppf(1 - 0.5 * alpha) * SE

    covered = (upper > truth) * (lower < truth)

    results_df = pd.DataFrame({'naive_pivot':pivots,
                               'naive_pvalue':pvalues,
                               'naive_coverage':covered,
                               'naive_length':upper - lower,
                               'naive_upper':upper,
                               'naive_lower':lower,
                               'variable':formula_terms,
                               })

    return beta_hat, target_cov, results_df

#### A parametric model will need something like this

class probit_step_learner(mixture_learner):

    def __init__(self,
                 algorithm, 
                 observed_selection,
                 target_cov,
                 X,
                 observed_MLE,
                 observed_Y):

        (self.algorithm,
         self.observed_outcome,
         self.target_cov,
         self.X,
         self.observed_MLE,
         self.observed_Y) = (algorithm,
                             observed_selection,
                             target_cov,
                             X,
                             observed_MLE,
                             observed_Y)

        n, p = X.shape
        var_select = ['X%d' % (i+1) in observed_selection for i in range(p)]
        self.X_select = np.hstack([np.ones((n, 1)), X[:,var_select]])

        self.observed_target = observed_MLE
        self._chol = np.linalg.cholesky(self.target_cov)
        self._beta_cov = self.target_cov

    def learning_proposal(self):
        """
        Return perturbed data and perturbed MLE.
        """

        n, s = self.X_select.shape

        beta_hat = self.observed_MLE

        perturbed_beta = beta_hat.copy()
        nidx = np.random.choice(np.arange(s), min(3, s), replace=False)
        for idx in nidx:
            scale = np.random.choice(self.scales, 1)
            perturbed_beta[idx] += (scale * np.random.standard_normal() *
                                    np.sqrt(self._beta_cov[idx, idx]))
        
        linpred = self.X_select.dot(perturbed_beta)
        prob = normal_dbn.cdf(linpred)
        perturbed_Y = np.random.binomial(1, prob)

        perturbed_MLE = probit_MLE(self.X, perturbed_Y, self.observed_outcome)[0]
        return perturbed_MLE, perturbed_Y

####

def simulate(n=500, p=10, alpha=0.1, B=2000):

    # description of statistical problem

    X = np.random.standard_normal((n, p))
    y = np.random.binomial(1, 0.5, size=(n,))
    truth = np.zeros(p+1)

    def algorithm(X, y):

        numpy2ri.activate()
        rpy.r.assign('X', X)
        rpy.r.assign('y', y)
        rpy.r('''
y = as.matrix(y)
D = data.frame(X, y)
glm(y ~ ., family=binomial(link='probit'), data=D)
M = glm(y ~ ., family=binomial(link='probit'), data=D)
M0 = glm(y ~ 1, family=binomial(link='probit'), data=D)
Mselect = step(M0, direction='forward', scope=list(upper=M, lower=M0), trace=FALSE)
selected_vars = names(coef(Mselect))
''')
        selected_vars = ' + '.join(sorted(list(rpy.r('selected_vars'))))
        selected_vars = selected_vars.replace('(Intercept)', '1')
        numpy2ri.deactivate()
        return tuple(selected_vars.split(' + '))

    # run selection algorithm

    selection_algorithm = functools.partial(algorithm, X)

    instance_hash = hashlib.md5()
    instance_hash.update(X.tobytes())
    instance_hash.update(y.tobytes())
    instance_hash.update(truth.tobytes())
    instance_id = instance_hash.hexdigest()

    observed_model = selection_algorithm(y)
    print(observed_model)

    proj_truth = np.zeros(len(observed_model)) # null simulation here
    MLE, target_cov, naive_df = probit_MLE(X, 
                                           y, 
                                           observed_model, 
                                           truth=proj_truth,
                                           alpha=alpha)

    (pivots, 
     covered, 
     lengths, 
     pvalues,
     lower,
     upper) = [], [], [], [], [], []

    targets = []

    s = len(observed_model)

    learner = probit_step_learner(selection_algorithm, 
                                  observed_model,
                                  target_cov,
                                  X,
                                  MLE,
                                  y)

    results = infer_general_target(observed_model,
                                   MLE,
                                   target_cov,
                                   learner,
                                   hypothesis=proj_truth,
                                   fit_probability=random_forest_fit_sk,
                                   fit_args={'n_estimators':5000},
                                   alpha=alpha,
                                   B=B)

    for result, true_target in zip(results, proj_truth):

        (pivot, 
         interval,
         pvalue,
         _) = result

        pvalues.append(pvalue)
        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])
        lower.append(interval[0])
        upper.append(interval[1])

    df = pd.DataFrame({'pivot':pivots,
                       'pvalue':pvalues,
                       'coverage':covered,
                       'length':lengths,
                       'upper':upper,
                       'lower':lower,
                       'target':proj_truth,
                       'variable':list(observed_model),
                       'id':[instance_id]*len(pivots),
                       })

    df = pd.merge(df, naive_df, on='variable')

    return df

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(2000):
        df = simulate(B=1000)
        csvfile = 'probit_step.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)


