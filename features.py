"""
This document contains the functions needed to extract the features from the raw file.

Author: Jacobo Fernandez Vargas
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import describe, mode
from time import time


def define_windows(data: NDArray, winLen: float, stride: int) -> NDArray:
    """
    Creates the index array for each window. It doesn't assume any fix sample rate for the data. However, for efficiency
    reasons, the maximum number of samples that will be evaluated for each window (to see which sample's time stamp
    is closet to the window length)is 30 * windowLen in seconds. This will work as long as the average sampling rate
    is not much higher than 30 Hz (according to the documentation is 20 Hz).
    :param data: Raw data as read from the csv file
    :param winLen: Window len in nanoseconds.
    :param stride: Every how many samples we want to create a window.
    :return windows: Nx2 Matrix containing the starting and ending indexes of each of the N windows.
    """
    maxStep = int(winLen / 1e9 * 20 * 1.5)
    minLen = int(winLen / 1e9 * 5)
    windows = []
    t = data[:, 2]
    st = np.arange(0, len(t) - maxStep, stride)
    for i in st:
        end = np.argmax((t[i: i+maxStep] - t[i]) > winLen)
        if end > minLen:
            windows.append([i, i + end])
    return np.array(windows)


def peak_detector_moving_thr(data: NDArray, thrMin: float=0.25, decay: float=1/20) -> NDArray:
    """
    Peak detection algorthim that uses a moving threshold to detect the peaks. The threshold decays in an linear
    way at a rate of decay, until detects a peak over the threshold. The minimum value of the threshold is the amplitude
    of the previous peak times thrMin. The algorithm may not work if the are peaks with negative values.
    :param data: Array of size N containing the time series data.
    :param thrMin: Minimum proportion of the peak for the threshold.
    :param decay: Rate at which the threshold reduces. It its highly dependent on the sample frequency, so it may
    require some tunning. Intuitively, if the decay is 1 / X, it means that it would take X steps to accept any peak.
    :return peaks: Array of size N with 0s where there were no peaks, and the amplitude of the peak where there was.
    """
    peaks = np.zeros(len(data))
    v = data[0]
    vmin = v * thrMin
    th = v
    peaks[0] = v
    for i in range(1, len(data) - 1):
        if data[i] > th and data[i] > data[i + 1] and data[i] > data[i - 1]:
            v = data[i]
            th = v
            vmin = v * thrMin
            peaks[i] = v
        else:
            th = np.max([th - v * decay, vmin])
    return peaks


def feature_extraction(data: NDArray, winLen: float=5e9, stride: int=1) -> tuple[NDArray, NDArray]:
    """
    This function extracts features from the given data.

    The features are:
    - Mean
    - Variance
    - Skewness
    - Kurtosis
    - Minimum value
    - Maximum value
    - Number of peaks
    - Mean peak amplitude

    :param data: The input data from which features are to be extracted. It is expected to be a numpy array.
    :param winLen: The length of the window in nanoseconds for which features are to be calculated. Default is 5e9.
    :param stride: The number of steps to move the window at each iteration. Default is 1.
    :return features: An Nx8 matrix containing the 8 features for each of the N windows.
    :return label:  An array of size N containing the activity for each of the windows.
    """
    acc = np.linalg.norm(data[:, -3:], axis=1)
    windows = define_windows(data, winLen, stride)
    peak = peak_detector_moving_thr(acc)
    # mean, std, skew, kurt, min, max, peakNum, peakAmp
    features = np.zeros((len(windows), 8))
    # User and Activity
    label = np.zeros((len(windows), 2)) - 1
    t0 = time()
    lw = len(windows)
    for i, (st, end) in enumerate(windows):
        if data[st, 0] == data[end, 0]:
            des = describe(acc[st: end])
            features[i, 0] = des.mean
            features[i, 1] = des.variance
            features[i, 2] = des.skewness
            features[i, 3] = des.kurtosis
            features[i, 4] = des.minmax[0]
            features[i, 5] = des.minmax[1]
            features[i, 6] = (peak[st: end] != 0).sum()
            features[i, 7] = peak[st: end].sum() / (features[i, 7] + 1e-12)

            label[i, 0] = data[st, 0]
            label[i, 1] = mode(data[st:end, 1]).mode
        if (i + 1) % 10000 == 0:
            t1 = time()
            dt = t1 - t0
            totTime = dt / i * lw
            print(f'Step {i+1} / {lw}, time: {dt}, estimated tot {totTime}, rem {totTime - dt} ')
    val = label[:, 0] != -1
    features = features[val]
    label = label[val]
    return features, label

