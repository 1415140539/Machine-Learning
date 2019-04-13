import os
import sys
import platform
import numpy as np
import sklearn.preprocessing as sp


def minmax(raw_sample,mi = 0,ma =1 ):
    copy_sample = raw_sample.copy()
    cols = copy_sample.shape[1]
    for col in range(cols):
        col_sample = copy_sample[:,col]
        min_sample = col_sample.min()
        max_sample = col_sample.max()
        k, b = np.linalg.lstsq(
            np.array([[min_sample,1],
                      [max_sample,1]]),
            np.array([mi,ma]))[0]
        col_sample *= k
        col_sample += b
    return np.round(copy_sample,8)
def main(argc, argv, environ):
    raw_samples = np.array([
        [3, -1.5,  2,   -5.4],
        [0,  4,   -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    samples = minmax(raw_samples, 0, 1)
    print(samples)
    mmx = sp.MinMaxScaler(feature_range=(0, 1))
    mmx_samples = mmx.fit_transform(raw_samples)
    print(mmx_samples)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))