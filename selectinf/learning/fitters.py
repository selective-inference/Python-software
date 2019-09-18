import uuid, functools

import numpy as np
from scipy.stats import norm as ndist
from sklearn import ensemble

def gbm_fit_sk(T, Y, **params):

    fitfns = []
    for j in range(Y.shape[1]):
        print('variable %d' % (j+1,))
        y = Y[:,j].astype(np.int)
        clf = ensemble.GradientBoostingClassifier(**params)
        clf.fit(T, y)

        def fit_fn(clf, t):
            return clf.predict_proba(t)[:,1]

        fitfns.append(functools.partial(fit_fn, clf))

    return fitfns

def random_forest_fit_sk(T, Y, **params):

    fitfns = []
    for j in range(Y.shape[1]):
        print('variable %d' % (j+1,))
        y = Y[:,j].astype(np.int)
        clf = ensemble.RandomForestClassifier(**params)
        clf.fit(T, y)

        def fit_fn(clf, t):
            return clf.predict_proba(t)[:,1]

        fitfns.append(functools.partial(fit_fn, clf))

    return fitfns

