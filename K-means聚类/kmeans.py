import os
import sys
import platform
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp


def read_data(filename):
    x = []
    with open(filename,"r") as f:
        for line in f.readlines():
            data = [float(substr) for substr in line[:-1].split(",")]
            x.append(data)
    return np.array(x)
def train_model(train_x):
    model = sc.KMeans(init ="k-means++",n_clusters= 4 , n_init=10)
    model.fit(train_x)
    return model
def pred_model(model,test_x):
    y = model.predict(test_x)
    return y
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 /255)
    mp.title("KMeans Classifier",fontsize = 24)
    mp.xlabel("X",fontsize = 16)
    mp.ylabel("Y",fontsize = 16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which = "both", right = True,top = True,
                   labelright = True, labelsize = 10)
def draw_grid(grid_x,grid_y):
    mp.pcolormesh(grid_x[0],grid_x[1],grid_y,cmap = "RdYlBu")
    mp.xlim(grid_x[0].min(),grid_x[0].max())
    mp.ylim(grid_x[1].min(),grid_x[1].max())
def draw_data(x,y):
    mp.scatter(x[:,0],x[:,1],c = y, marker = "o", s= 80)
def draw_centers(x):
    mp.scatter(x[:,0],x[:,1],c = "black",marker = "+",s = 300,label ="center")
    mp.legend
def show_chart():
    mng = mp.get_current_fig_manager()
    if "window" in platform.system():
        mng.window.state("zoomed")
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
def main(argc, argv, envir):
    x = read_data("multiple.txt")
    model = train_model(x)
    l,e,s = x[:,0].min()-1 , x[:,0].max() +1 , 0.005
    b,t,v = x[:,1].min()-1,x[:,1].max() +1 ,0.005
    grid_x = np.meshgrid(np.arange(l,e,s),
                         np.arange(b,t,v))
    grid_y = pred_model(model,np.c_[grid_x[0].ravel(),
                                    grid_x[1].ravel()]
                        ).reshape(grid_x[0].shape)
    pred_y = pred_model(model,x)
    init_chart()
    draw_grid(grid_x,grid_y)
    draw_data(x,pred_y)
    print(model.cluster_centers_)
    draw_centers(model.cluster_centers_)
    show_chart()
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))