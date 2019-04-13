import os
import sys
import platform
import numpy as np
import sklearn.preprocessing as sp


def main(argc,argv,envir):
    raw_labels = np.array([
        'audi', 'ford', 'audi', 'toyota', 'ford', 'bmw',
        'toyota', 'ford', 'audi'])
    print(raw_labels)
    encoder = sp.LabelEncoder()
    encode = encoder.fit_transform(raw_labels)
    label = encoder.inverse_transform(encode)
    print(encode)
    print(label)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))