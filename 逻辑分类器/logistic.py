import os
import sys
import platform
import numpy as np
import sklearn.linear_model as sl
import sklearn.model_selection as sm
import matplotlib.pyplot as mp


def make_data():
    x = np.array([
        [4, 7],
        [3.5, 8],
        [3.1, 6.2],
        [0.5, 1],
        [1, 2],
        [1.2, 1.9],
        [6, 2],
        [5.7, 1.5],
        [5.4, 2.2]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return x, y
def train_model(x,y):
    model = sl.LogisticRegression(solver='liblinear',C = 80)
    model.fit(x,y)
    return model
def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Logistic Classifier', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(axis='y', linestyle=':')
def draw_bk_chart(x,y):
    mp.pcolormesh(x[0],x[1], y,cmap = "brg")
    mp.xlim(x[0].min(), x[0].max())
    mp.ylim(x[1].min(), x[1].max())
def draw_chart(x,y):
    mp.scatter(x[:,0],x[:,1],c = y, cmap = "RdYlBu",s = 80)
def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
def main(argc, argv, envir):

    x, y = make_data()
    model = train_model(x,y)
    r,e,s =x[:,0].min()-1, x[:,0].max()+1,0.005
    b,v,t = x[:,1].min()-1,x[:,1].max()+1,0.005
    grid_x = np.meshgrid(np.arange(r,e,s),
                         np.arange(b,v,t))
    grid_y = pred_model(model,
                        np.c_[grid_x[0].ravel()
                        ,grid_x[1].ravel()]
                        ).reshape(grid_x[0].shape)
    init_chart()
    draw_bk_chart(grid_x,grid_y)
    draw_chart(x,y)
    show_chart()
    return 0

if __name__ =="__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))