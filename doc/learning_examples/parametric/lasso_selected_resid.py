import functools, hashlib

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance

from selection.learning.learners import mixture_learner
from selection.learning.utils import naive_partial_model_inference, pivot_plot
from selection.learning.core import keras_fit, infer_general_target

#### A parametric model will need something like this

class gaussian_OLS_learner(mixture_learner):

    def __init__(self,
                 algorithm, 
                 observed_selection,
                 X_select,
                 observed_MLE,
                 observed_Y, 
                 dispersion):

        (self.algorithm,
         self.observed_outcome,
         self.X_select,
         self.observed_MLE,
         self.observed_Y) = (algorithm,
                             observed_selection,
                             X_select,
                             observed_MLE,
                             observed_Y)

        self.observed_target = observed_MLE

        self._dispersion = dispersion
        gram_matrix = X_select.T.dot(X_select)
        self._chol = (np.linalg.cholesky(np.linalg.inv(gram_matrix)) * 
                      np.sqrt(self._dispersion))
        n, p = X_select.shape
        self._Xinv = np.linalg.pinv(X_select)
        self._beta_cov = self._Xinv.dot(self._Xinv.T) * self._dispersion
        self._resid = observed_Y - X_select.dot(self._Xinv.dot(observed_Y))

    def learning_proposal(self):
        """
        Return perturbed data and perturbed MLE.
        """

        n, s = self.X_select.shape

        beta_hat = self.observed_MLE

        scale = np.random.choice(self.scales, 1)
        idx = np.random.choice(np.arange(s), 1)
        perturbed_beta = beta_hat * 1.
        perturbed_beta[idx] += (scale * np.random.standard_normal() *
                                np.sqrt(self._beta_cov[idx, idx]))
        
        perturbed_Y = self.X_select.dot(perturbed_beta) + self._resid
        return perturbed_beta, perturbed_Y

####

def simulate(n=500, p=30, s=5, signal=(0.5, 1.), sigma=2, alpha=0.1, B=2000):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False)[:3]

    def algorithm(lam, X, y):

        n, p = X.shape
        success = np.zeros(p)

        loss = rr.quadratic_loss((p,), Q=X.T.dot(X))
        pen = rr.l1norm(p, lagrange=lam)

        S = -X.T.dot(y)
        loss.quadratic = rr.identity_quadratic(0, 0, S, 0)
        problem = rr.simple_problem(loss, pen)
        soln = problem.solve(max_its=100, tol=1.e-10)
        success += soln != 0
        return set(np.nonzero(success)[0])

    # run selection algorithm

    lam = 2. * np.sqrt(n)
    selection_algorithm = functools.partial(algorithm, lam, X)

    instance_hash = hashlib.md5()
    instance_hash.update(X.tobytes())
    instance_hash.update(y.tobytes())
    instance_hash.update(truth.tobytes())
    instance_id = instance_hash.hexdigest()

    observed_tuple = selection_algorithm(y)

    (pivots, 
     covered, 
     lengths, 
     pvalues,
     lower,
     upper) = [], [], [], [], [], []

    targets = []

    if len(observed_tuple) > 0:

        s = len(observed_tuple)

        X_select = X[:, list(observed_tuple)]
        Xpi = np.linalg.pinv(X_select)

        final_target = Xpi.dot(X.dot(truth))
        observed_target = Xpi.dot(y)

        resid = y - X_select.dot(observed_target)
        dispersion = np.linalg.norm(resid)**2 / (n-s)

        target_cov = Xpi.dot(Xpi.T) * dispersion

        MLE = observed_target

        learner = gaussian_OLS_learner(selection_algorithm, 
                                       observed_tuple,
                                       X_select,
                                       MLE,
                                       y,
                                       dispersion)

        print(observed_tuple)
        results = infer_general_target(observed_tuple,
                                       MLE,
                                       target_cov,
                                       learner,
                                       hypothesis=final_target,
                                       fit_probability=keras_fit,
                                       fit_args={'epochs':30, 
                                                 'sizes':[100]*5, 
                                                 'dropout':0., 
                                                 'activation':'relu'},
                                       alpha=alpha,
                                       B=B)

        for result, true_target in zip(results, final_target):
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

    if len(observed_tuple) > 0:

        df = pd.DataFrame({'pivot':pivots,
                           'pvalue':pvalues,
                           'coverage':covered,
                           'length':lengths,
                           'upper':upper,
                           'lower':lower,
                           'target':final_target,
                           'variable':list(observed_tuple),
                           'id':[instance_id]*len(pivots),
                           })

        naive = True # report naive intervals as well?

        if naive:
            naive_df = naive_partial_model_inference(X,
                                                     y,
                                                     dispersion,
                                                     truth,
                                                     observed_tuple,
                                                     alpha=alpha)
            df = pd.merge(df, naive_df, on='variable')

        return df

if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import pandas as pd

    for i in range(2000):
        df = simulate(B=15000)
        csvfile = 'lasso_selected_resid.csv'
        outbase = csvfile[:-4]

        if df is not None and i > 0:

            try: # concatenate to disk
                df = pd.concat([df, pd.read_csv(csvfile)])
            except FileNotFoundError:
                pass
            df.to_csv(csvfile, index=False)

            if len(df['pivot']) > 0:
                pivot_ax, length_ax = pivot_plot(df, outbase)


