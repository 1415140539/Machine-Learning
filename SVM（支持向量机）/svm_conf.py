import os
import sys
import platform
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp


def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])
    return np.array(x), np.array(y, dtype=int)


def train_model(x, y):
    model = svm.SVC(kernel='rbf', C=600, gamma=0.01,
                    probability=True)
    model.fit(x, y)
    return model


def pred_model(model, x):
    y = model.predict(x)
    return y


def eval_cr(y, pred_y):
    cr = sm.classification_report(y, pred_y)
    print(cr)


def make_data():
    x = np.array([
        [2, 1.5],
        [8, 9],
        [4.8, 5.2],
        [4, 4],
        [2.5, 7],
        [7.6, 2],
        [5.4, 5.9]])
    return x


def eval_cp(model, x):
    cp = model.predict_proba(x)
    return cp


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('SVM RBF Classifier', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)


def draw_grid(grid_x, grid_y):
    mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
    mp.xlim(grid_x[0].min(), grid_x[0].max())
    mp.ylim(grid_x[1].min(), grid_x[1].max())


def draw_data(x, y):
    C0, C1 = y == 0, y == 1
    mp.scatter(x[C0][:, 0], x[C0][:, 1], c='orangered', s=80)
    mp.scatter(x[C1][:, 0], x[C1][:, 1], c='limegreen', s=80)


def draw_cp(cp_x, cp_y, cp):
    C0, C1 = cp_y == 0, cp_y == 1
    mp.scatter(cp_x[C0][:, 0], cp_x[C0][:, 1], marker='D',
               c='dodgerblue', s=80)
    mp.scatter(cp_x[C1][:, 0], cp_x[C1][:, 1], marker='D',
               c='deeppink', s=80)
    for i in range(len(cp[C0])):
        mp.annotate(
            '{}% {}%'.format(round(cp[C0][:, 0][i] * 100, 2),
                             round(cp[C0][:, 1][i] * 100, 2)),
            xy=(cp_x[C0][:, 0][i], cp_x[C0][:, 1][i]),
            xytext=(12, -12),
            textcoords='offset points',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=9,
            bbox={'boxstyle': 'round, pad=0.6',
                  'fc': 'deepskyblue', 'alpha': 0.8})
    for i in range(len(cp[C1])):
        mp.annotate(
            '{}% {}%'.format(round(cp[C1][:, 0][i] * 100, 2),
                             round(cp[C1][:, 1][i] * 100, 2)),
            xy=(cp_x[C1][:, 0][i], cp_x[C1][:, 1][i]),
            xytext=(12, -12),
            textcoords='offset points',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=9,
            bbox={'boxstyle': 'round, pad=0.6',
                  'fc': 'violet', 'alpha': 0.8})


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    x, y = read_data('binary.txt')
    l, r, h = x[:, 0].min() - 1, x[:, 0].max() + 1, 0.005
    b, t, v = x[:, 1].min() - 1, x[:, 1].max() + 1, 0.005
    train_x, test_x, train_y, test_y = ms.train_test_split(
        x, y, test_size=0.25, random_state=5)
    model = train_model(train_x, train_y)
    grid_x = np.meshgrid(np.arange(l, r, h),
                         np.arange(b, t, v))
    grid_y = pred_model(
        model,
        np.c_[grid_x[0].ravel(),
              grid_x[1].ravel()]).reshape(grid_x[0].shape)
    pred_test_y = pred_model(model, test_x)
    eval_cr(test_y, pred_test_y)
    cp_x = make_data()
    cp_y = pred_model(model, cp_x)
    cp = eval_cp(model, cp_x)
    print(cp)
    init_chart()
    draw_grid(grid_x, grid_y)
    draw_data(x, y)
    draw_cp(cp_x, cp_y, cp)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
