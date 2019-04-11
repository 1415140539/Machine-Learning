import os
import sys
import platform
import numpy as np
import sklearn.cluster as sc
import sklearn.metrics as ms
import matplotlib.pyplot as mp


def read_data(filename):
    x = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data)
    return np.array(x)


def train_model(x):
    epsilons = np.linspace(0.3, 1.2, 10)
    scores = []
    models = []
    for epsilon in epsilons:
        model = sc.DBSCAN(
            eps=epsilon, min_samples=5).fit(x)
        scores.append(ms.silhouette_score(
            x, model.labels_, sample_size=len(x),
            metric='euclidean'))
        models.append(model)
    scores = np.array(scores)
    best_index = scores.argmax()
    best_epsilon = epsilons[best_index]
    best_score = scores[best_index]
    best_model = models[best_index]
    print(best_epsilon, best_score)
    return best_model


def pred_model(model, x):
    y = model.fit_predict(x)
    core_mask = np.zeros(len(x), dtype=bool)
    core_mask[model.core_sample_indices_] = True
    offset_mask = model.labels_ == -1
    periphery_mask = ~(core_mask | offset_mask)
    return y, core_mask, offset_mask, periphery_mask


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('DBSCAN Cluster', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y, core_mask, offset_mask, periphery_mask):
    labels = set(y)
    cs = mp.get_cmap('brg', len(labels))(range(len(labels)))
    mp.scatter(x[core_mask][:, 0], x[core_mask][:, 1],
               c=cs[y[core_mask]], s=80, label='Core')
    mp.scatter(x[offset_mask][:, 0], x[offset_mask][:, 1],
               marker='x', c=cs[y[offset_mask]], s=80,
               label='Offset')
    mp.scatter(x[periphery_mask][:, 0], x[periphery_mask][:, 1],
               edgecolor=cs[y[periphery_mask]], s=80,
               facecolor='none', label='Periphery')
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    x = read_data('perf.txt')
    model = train_model(x)
    pred_y, core_mask, offset_mask, periphery_mask = pred_model(
        model, x)
    init_chart()
    draw_chart(x, pred_y, core_mask, offset_mask, periphery_mask)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
