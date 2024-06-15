import numpy as np
from numpy.typing import NDArray
from scipy.stats import describe, mode
from time import time

def define_windows(data: NDArray, winLen: float, stride: int) -> NDArray:
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

