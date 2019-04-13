import os
import sys
import numpy as np
import sklearn.preprocessing as sp

def binarzie(raw_samples, threshold):
    col_samples = raw_samples.copy()
    col_samples[col_samples <= threshold] = 0
    col_samples[col_samples > threshold] = 1
    return  col_samples

raw_samples = np.array([
    [3, -1.5,  2,   -5.4],
    [0,  4,   -0.3,  2.1],
    [1,  3.3, -1.9, -4.3]])
def main(argc, argv, envp):
    samples = binarzie(raw_samples,2)
    print(samples)
if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))