import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV


class ThressholdScaler(BaseEstimator, TransformerMixin):
    def __init__(self, qoffset=1.5):
        self.qoffset = qoffset
        self.lower = None
        self.upper = None

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower = q1 - self.qoffset * iqr
        self.upper = q3 + self.qoffset * iqr
        return self

    def transform(self, X, y=None):
        nFeat = len(self.upper)
        for i in range(nFeat):
            X[X[:, i] > self.upper[i], i] = self.upper[i]
            X[X[:, i] < self.lower[i], i] = self.lower[i]
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


def data_split(x, y):
    us = np.unique(y[:, 0])
    ua = np.unique(y[:, 1])
    xTst, xTr = [], []
    yTst, yTr = [], []
    trainPct = 0.9
    for i, s in enumerate(us):
        for j, a in enumerate(ua):
            val = np.logical_and(y[:, 0] == s, y[:, 1] == a)
            xTemp = x[val]
            yTemp = y[val, 1]
            idx = int(len(yTemp) * trainPct)
            xTst.append(xTemp[idx:])
            xTr.append(xTemp[:idx])
            yTst.append(yTemp[idx:])
            yTr.append(yTemp[:idx])
    xTst = np.concatenate(xTst)
    xTr = np.concatenate(xTr)
    yTst = np.concatenate(yTst)
    yTr = np.concatenate(yTr)
    return xTst, xTr, yTst, yTr


def hyper_parameter_search(x, y):
    cv = StratifiedKFold(shuffle=True)
    ts = ThressholdScaler()
    mdl = RandomForestClassifier()
    pipe = Pipeline([('scaler', ts), ('model', mdl)])
    param_grid = {
        'model__criterion': ['gini', 'entropy', 'log_loss'],
        'model__n_estimators': [5, 10, 100],
        'model__max_depth': [3, 7, 12]
    }
    kappa = mt.make_scorer(mt.cohen_kappa_score)
    metrics = {'acc': 'balanced_accuracy',
               'f1': 'f1_macro',
               'kappa': kappa}
    search = GridSearchCV(pipe, param_grid, scoring=metrics, n_jobs=-1, cv=3, refit='acc')
    scores = cross_validate(search, x, y, scoring=metrics, cv=cv, return_estimator=True)
    return scores