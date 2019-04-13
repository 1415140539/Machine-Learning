import os
import sys
import platform
import numpy as np
import sklearn.neighbors as sn
import sklearn.metrics as sm
import matplotlib.pyplot as mp


def read_data(filename):
    x, y = [], []
    with open(filename,"r") as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(",")]
            x.append(data[:-1])
            y.append(data[-1])
    return np.array(x), np.array(y)
def train_model(x,y):
    model = sn.KNeighborsClassifier(n_neighbors=10,
                                    weights="distance")
    model.fit(x,y)
    return model
def pred_model(model, x):
    pred_y = model.predict(x)
    return pred_y
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240/255)
    mp.title("Knn Classifier", fontsize = 24)
    mp.xlabel("X",fontsize = 16)
    mp.ylabel("Y",fontsize = 16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which = "both", right = True,
                   top = True, labelright = True,
                   labelsize = 10)
def draw_grid(x,y):
    mp.pcolormesh(x[0],x[1],y,cmap = "RdYlBu")
    mp.xlim(x[0].min(),x[0].max())
    mp.ylim(x[1].min(),x[1].max())
def draw_point(x,y,test_x,pred_y,nn_indices):
    classes = np.unique(y)
    classes.sort()
    cs = mp.get_cmap("jet",len(classes))(classes)
    mp.scatter(x[:,0],x[:,1],c = cs[y], s= 80)
    mp.scatter(test_x[:,0],test_x[:,1],c = cs[pred_y], marker='D',s = 120)
    for i, nn_index in enumerate(nn_indices):
        mp.scatter(x[nn_index,0],x[nn_index,1],marker = "D",
                   edgecolor=cs[np.ones_like(nn_index) * pred_y[i]],
                   facecolor='none', linewidth=2, s=180)
    mp.show()
def test_data():
    x = np.array([
        [2.2, 6.2],
        [3.6, 1.8],
        [4.5, 3.6]])
    return x
def get_nn_indices(model,x):
    nn_distiancs,nn_indices = model.kneighbors(x)
    return nn_distiancs, nn_indices
def main(argc,argv,envir):
    x, y = read_data("knn.txt")
    model = train_model(x, y)
    l,e,s = x[:,0].min()-1, x[:,0].max() +1 ,0.005
    b,t,v = x[:,1].min()-1, x[:,1].max() +1 , 0.005
    grid_x = np.meshgrid(np.arange(l,e,s),
                         np.arange(b,t,v))
    grid_y = pred_model(model,
                        np.c_[grid_x[0].ravel(),grid_x[1].ravel()]
                        ).reshape(grid_x[0].shape)
    test_x = test_data()
    nn_distiance,nn_indices = get_nn_indices(model,test_x)
    print(nn_indices)
    pred_y = pred_model(model,test_x)
    init_chart()
    draw_grid(grid_x,grid_y)
    draw_point(x,y.astype(int),test_x,pred_y.astype(int),nn_indices)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))
