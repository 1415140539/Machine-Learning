import os
import sys
import platform
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms
import matplotlib.pyplot as mp


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1].split(','))
    data = np.array(data).T
    encoders, x = [], []
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data) - 1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return x, y


def train_model_estimator(max_depth):
    model = se.RandomForestClassifier(
        max_depth=max_depth, random_state=7)
    return model


def eval_vc_estimator(model, x, y, n_estimators):
    train_scores, test_scores = ms.validation_curve(
        model, x, y, 'n_estimators', n_estimators, cv=5)
    print(train_scores)
    print(test_scores)
    return train_scores, test_scores


def train_model_max_depth(n_estimators):
    model = se.RandomForestClassifier(
        n_estimators=n_estimators, random_state=7)
    return model


def eval_vc_max_depth(model, x, y, max_depth):
    train_scores, test_scores = ms.validation_curve(
        model, x, y, 'max_depth', max_depth, cv=5)
    print(train_scores)
    print(test_scores)
    return train_scores, test_scores


def init_estimator():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Training Curve On Estimator', fontsize=20)
    mp.xlabel('Number Of Estimators', fontsize=14)
    mp.ylabel('Accuracy', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(20))
    ax.yaxis.set_major_locator(mp.MultipleLocator(.1))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_estimator(n_estimators, train_score_estimator):
    mp.plot(n_estimators,
            train_score_estimator.mean(axis=1) * 100,
            'o-', c='dodgerblue', label='Train Score')
    mp.legend()


def init_max_depth():
    mp.subplot(122)
    mp.title('Training Curve On Max Depth', fontsize=20)
    mp.xlabel('Max Depth', fontsize=14)
    mp.ylabel('Accuracy', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator(5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_max_depth(max_depth, train_score_max_depth):
    mp.plot(max_depth,
            train_score_max_depth.mean(axis=1) * 100,
            'o-', c='orangered', label='Train Score')
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    x, y = read_data('car.txt')
    model_estimator = train_model_estimator(4)
    n_estimators = np.linspace(20, 200, 10).astype(int)
    train_score_estimator, test_score_estimator = \
        eval_vc_estimator(model_estimator, x, y, n_estimators)
    model_max_depth = train_model_max_depth(20)
    max_depth = np.linspace(1, 10, 10).astype(int)
    train_score_max_depth, test_score_max_depth = \
        eval_vc_max_depth(model_max_depth, x, y, max_depth)
    init_estimator()
    draw_estimator(n_estimators, train_score_estimator)
    init_max_depth()
    draw_max_depth(max_depth, train_score_max_depth)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
