import os
import sys
import platform
import numpy as np
import scipy.misc as sm
import sklearn.cluster as sc
import matplotlib.pyplot as mp


def train_model(n_clusters, x):
    model = sc.KMeans(n_init=4, n_clusters=n_clusters,
                      random_state=5)
    model.fit(x)
    return model


def load_image(image_file):
    return sm.imread(image_file, True).astype(np.uint8)


def compress_image(image, bpp):
    n_clusters = np.power(2, bpp)
    print(n_clusters)
    x = image.reshape((-1, 1))
    print(x)
    model = train_model(n_clusters, x)
    y = model.labels_
    print(y)
    centers = model.cluster_centers_.squeeze()
    print(centers)
    z = centers[y]
    print(z)
    return z.reshape(image.shape)


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Compress Image', fontsize=20)
    mp.axis('off')


def draw_chart(image):
    mp.imshow(image, cmap='gray')


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    image = load_image('flower.jpg')
    compressed_image = compress_image(image, 1)
    init_chart()
    draw_chart(compressed_image)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
