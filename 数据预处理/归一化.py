import os
import sys
import platform
import numpy as np
import sklearn.preprocessing as sp


def normalize(raw_samples):
    copy_samples = raw_samples.copy()
    rows = copy_samples.shape[0]
    for row in range(rows):
        row_sample = copy_samples[row]
        row_abs = abs(row_sample)
        row_abs_sum = row_abs.sum()
        row_sample /= row_abs_sum
    return copy_samples
def main(argc, argv, envir):
    raw_samples = np.array([
        [3, -1.5,  2,   -5.4],
        [0,  4,   -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    samples = normalize(raw_samples)
    print(samples)
    samples_1 = sp.normalize(raw_samples,"l1")
    print(samples_1)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv, os.environ))