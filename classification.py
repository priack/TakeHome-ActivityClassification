import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
import plotly.graph_objects as go
import copy


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
    metrics = {'acc': 'balanced_accuracy'}
    search = GridSearchCV(pipe, param_grid, scoring=metrics, n_jobs=-1, cv=3, refit='acc')
    scores = cross_validate(search, x, y, scoring=metrics, cv=cv, return_estimator=True)
    return scores


def train_model(x, y, params):
    ts = ThressholdScaler()
    mdl = RandomForestClassifier(**params)
    pipe = Pipeline([('scaler', ts), ('model', mdl)])
    pipe.fit(x, y)
    return pipe


def test_model(mdl, x, y):
    pred = mdl.predict(x)
    cm = mt.confusion_matrix(y, pred)
    acc = mt.balanced_accuracy_score(y, pred)
    f1 = mt.f1_score(y, pred, average='macro')
    kappa = mt.cohen_kappa_score(y, pred)
    return cm, acc, f1, kappa


def createConfusionMatrix(cm, labels, normalise: bool=True) -> go.Figure:
    z = copy.copy(cm)
    nrow = len(z)
    if normalise:
        for i in range(nrow):
            z[i, :] = cm[i, :] / np.sum(cm[i, :])
        print(z)
    text = [[f'{z[i,j]:.2f}' for j in range(nrow)] for i in range(nrow-1, -1, -1)]
    fig = go.Figure(go.Heatmap(z=z[::-1], y=labels[::-1], x=labels, text=text, texttemplate="%{text}", showscale=False,
                     colorscale='Greys'))
    fig.show()
    return fig