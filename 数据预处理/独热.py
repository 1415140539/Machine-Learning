import os
import sys
import platform
import numpy as np
import sklearn.preprocessing as sp


def deal_with_ohe(raw_sample):
    # --------------------#
    #    10   100  0001   #
    #    01   010  1000   #
    #    10   001  0100   #
    #    01   100  0010   #
    # --------------------#
    ohe_samples = []
    copy_sample = raw_sample
    cols = copy_sample.shape[1]
    for col in range(cols):
        col_sample = copy_sample[:,col]
        type = np.unique(col_sample).size
        ohe = []
        for raw  in  col_sample:
            sample = np.zeros(type)
            sample[raw] = 1
            ohe.append(sample)
        ohe_samples.append(ohe)
    print(np.array(ohe_samples).T)
def main(argc, argv, envir):
    raw_samples = np.array([
        [0, 0, 3],
        [1, 1, 0],
        [0, 2, 1],
        [1, 0, 2]])
    deal_with_ohe(raw_samples)
    ohe = sp.OneHotEncoder(sparse=False, dtype=int)
    ohe_samples = ohe.fit_transform(raw_samples)
    print(ohe_samples)

    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv, os.environ))