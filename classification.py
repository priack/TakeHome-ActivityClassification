"""
This document contains the functions needed to perfomr the classification task.

Author: Jacobo Fernandez Vargas
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
import plotly.graph_objects as go
import copy


class ThressholdScaler(BaseEstimator, TransformerMixin):
    def __init__(self, qoffset=1.5):
        """
        Threshold scaler based on Tukey's Fence algorithm. This class is to be used as part of a sklearn pipeline.
        :param qoffset: Factor to multiply the inter-quartile distance to consider a value outlier.
        """
        self.qoffset = qoffset
        self.lower = None
        self.upper = None

    def fit(self, X, y=None):
        """
        Calculates the maximum and minimum value for each feature.
        """
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower = q1 - self.qoffset * iqr
        self.upper = q3 + self.qoffset * iqr
        return self

    def transform(self, X, y=None):
        """
        Clips each feature to the maximum and minimum value previously calculated.
        """
        nFeat = len(self.upper)
        for i in range(nFeat):
            X[:, i] = np.clip(X[:, i], self.lower[i], self.upper[i])
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


def data_split(x: NDArray, y:NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Splits the data into training and testing data sets with a 9/1 proportion. It does so maintaing the temporal
    coherence of the features, as well as the distribution of the labels across subjects and activities.
    :param x: Features
    :param y: Labels
    :return xTst: Test features
    :return xTr: Training features
    :return yTst: Test labels
    :return yTr: Training labels
    """
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


def hyper_parameter_search(x: NDArray, y: NDArray) ->dict:
    """
    This function performs a hyperparameter search for a RandomForestClassifier using GridSearchCV.

    :param x: A numpy array representing the input data for training the model.
    :param y: A numpy array representing the actual labels for the input data.
    :return: A dictionary containing the cross-validation scores and estimators for each fold of the StratifiedKFold.
    """
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


def train_model(x: NDArray, y: NDArray, params: dict) -> Pipeline:
    """
    This function trains a model using the given data and parameters.

    :param x: A numpy array representing the input data for training the model.
    :param y: A numpy array representing the actual labels for the input data.
    :param params: A dictionary containing the parameters for the RandomForestClassifier.
    :return: A trained Pipeline object which includes a ThressholdScaler and a RandomForestClassifier.
    """
    ts = ThressholdScaler()
    mdl = RandomForestClassifier(**params)
    pipe = Pipeline([('scaler', ts), ('model', mdl)])
    pipe.fit(x, y)
    return pipe


def test_model(mdl, x: NDArray, y: NDArray) -> tuple[NDArray, float, float, float]:
    """
    This function tests a model by making predictions on the given data and calculating various metrics.
    :param mdl: The model to be tested.
    :param x: A numpy array representing the input data for testing the model.
    :param y: A numpy array representing the actual labels for the input data.
    :return: A tuple containing the confusion matrix, balanced accuracy score, macro F1 score, and Cohen's kappa score.
    """
    pred = mdl.predict(x)
    cm = mt.confusion_matrix(y, pred)
    acc = mt.balanced_accuracy_score(y, pred)
    f1 = mt.f1_score(y, pred, average='macro')
    kappa = mt.cohen_kappa_score(y, pred)
    return cm, acc, f1, kappa


def createConfusionMatrix(cm: NDArray, labels: tuple[str], normalise: bool=True) -> go.Figure:
    """
    This function creates a confusion matrix and visualizes it using a heatmap.
    :param cm: A numpy array representing the confusion matrix.
    :param labels: A tuple of strings representing the labels for the confusion matrix.
    :param normalise: A boolean value indicating whether to normalize the confusion matrix. Default is True.
    :return: A plotly.graph_objects.Figure object representing the heatmap of the confusion matrix.
    """
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