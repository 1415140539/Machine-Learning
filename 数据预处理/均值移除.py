import os
import sys
import platform
import numpy as np
import sklearn.preprocessing as sp

def std_scale(raw_samples):
    copy_samples = raw_samples.copy()
    cols = copy_samples.shape[1]
    for col in range(cols):
        col_sample = copy_samples[:,col]
        samples_mean = col_sample.mean()
        col_sample -= samples_mean
        sample_std = col_sample.std()
        col_sample /= sample_std
    return copy_samples
def main(argc,argv,envir):
    raw_samples = np.array([
        [3, -1.5,  2,   -5.4],
        [0,  4,   -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    print(std_scale(raw_samples).mean(axis = 0))
    std_samples = sp.scale(raw_samples)
    std_means = std_samples.mean(axis=0)
    std_stds = std_samples.std(axis=0)
    print(std_means)
    print(std_stds)
    return 0
if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))