"""
This document contains the main function to load the data, train a model, and plot a confusion matrix.
"""

import numpy as np
from io_utils import read_data
from features import feature_extraction
from classification import hyper_parameter_search, data_split, train_model, test_model, createConfusionMatrix

if __name__ == '__main__':
    data = read_data()
    winLen = 5e9  # 5s windows
    stride = 25 # If 20 Hz, this would be 1.25 s, but I have observed that data is not regularly at 20 Hz
    features, label = feature_extraction(data, winLen, stride)
    us = np.unique(label[:, 0])
    ua = np.unique(label[:, 1])
    dataDist = np.zeros((len(us), len(ua)))
    for i, s in enumerate(us):
        for j, a in enumerate(ua):
            val = np.logical_and(label[:, 0] == s, label[:, 1] == a)
            dataDist[i, j] = np.mean(val)

    xTst, xTr, yTst, yTr = data_split(features, label)
    scores = hyper_parameter_search(xTr, yTr)
    searchScore = np.zeros((3, 3, 3))
    for i in range(5):
        searchScore += scores['estimator'][i].cv_results_['mean_test_acc'].reshape(3, 3, 3) / 5

    # From this we get that the depth and number of estimators are quite relevant for this task, but not so much the
    # criterion. So we select entropy, 100, 12
    params = {'criterion': 'entropy', 'n_estimators': 100, 'max_depth': 12}
    mdl = train_model(xTr, yTr, params)
    cm, acc, f1, kappa = test_model(mdl, xTst, yTst)
    labels = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    cmf = createConfusionMatrix(cm, labels)


