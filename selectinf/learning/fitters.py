import uuid, functools

import numpy as np
from scipy.stats import norm as ndist
from sklearn import ensemble

def gbm_fit_sk(T, Y, **params):

    fitfns = []
    for j in range(Y.shape[1]):
        y = Y[:,j].astype(np.int)
        if len(np.unique(y)) > 1:
            clf = ensemble.GradientBoostingClassifier(**params)
            clf.fit(T, y)

            def fit_fn(clf, t):
                return clf.predict_proba(t)[:,1]
            fit_fn = functools.partial(fit_fn, clf)
        else:
            fit_fn = lambda t: np.atleast_1d(np.ones(t.shape[0]))
        fitfns.append(fit_fn)

    return fitfns

def random_forest_fit_sk(T, Y, **params):

    fitfns = []
    for j in range(Y.shape[1]):
        y = Y[:,j].astype(np.int)
        clf = ensemble.RandomForestClassifier(**params)
        clf.fit(T, y)

        def fit_fn(clf, t):
            return clf.predict_proba(t)[:,1]

        fitfns.append(functools.partial(fit_fn, clf))

    return fitfns

