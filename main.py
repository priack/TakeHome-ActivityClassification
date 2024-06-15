import numpy as np
from io_utils import read_data
from features import feature_extraction
from classification import hyper_parameter_search, data_split

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