import os
import sys
import platform
import numpy as np
import sklearn.neighbors as sn
import matplotlib.pyplot as mp


def make_data():
    x = 10 * np.random.rand(100, 1) - 5
    y = np.sinc(x).ravel()
    y += 0.2 * (0.5 - np.random.rand(y.size))
    return x, y


def train_model(x, y):
    model = sn.KNeighborsRegressor(n_neighbors=10,
                                   weights='distance')
    model.fit(x, y)
    return model


def pred_model(model, x):
    y = model.predict(x)
    return y


def test_data():
    x = np.linspace(-5, 5, 10000).reshape(-1, 1)
    return x


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('KNN Regressor', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.1))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_train(train_x, train_y):
    sorted_indices = train_x.ravel().argsort()
    mp.plot(train_x[sorted_indices], train_y[sorted_indices],
            'o-', c='dodgerblue', label='Training')


def draw_test(test_x, pred_test_y):
    mp.plot(test_x, pred_test_y, c='orangered', label='Testing')
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    train_x, train_y = make_data()
    model = train_model(train_x, train_y)
    test_x = test_data()
    pred_test_y = pred_model(model, test_x)
    init_chart()
    draw_train(train_x, train_y)
    draw_test(test_x, pred_test_y)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
