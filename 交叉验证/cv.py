import os
import sys
import platform
import numpy as np
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp


def read_data(filename):
    x, y = [], []
    with open(filename,"r") as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])
    return np.array(x), np.array(y)
def train_model(x,y):
    model = nb.GaussianNB()
    model.fit(x,y)
    return model
def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 /255)
    mp.title("Naive Bayes", fontsize = 24)
    mp.xlabel("x", fontsize = 16)
    mp.ylabel("y",fontsize = 16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))

    mp.tick_params(which = "both", top = True, right = True,
                   labelright = True, labelsize = 16)
    mp.grid(linestyle = ":")
def draw_point(grid_x,grid_y):
    mp.pcolormesh(grid_x[0],grid_x[1],grid_y, cmap = "brg")
    mp.xlim(grid_x[0].min(),grid_x[0].max())
    mp.ylim(grid_x[1].min(),grid_x[1].max())
def draw_data(x,y,pred_y):
    mp.scatter(x[:,0],x[:,1],c = y,marker = "D", cmap = "RdYlBu",s = 80)
    mp.scatter(x[:, 0], x[:, 1], c=pred_y, marker="X", cmap="RdYlBu", s=80)

def draw_train(x,y):
    mp.scatter(x[:,0],x[:,1],c = y,cmap="RdYlBu",s = 80)

def show_chart():
    mng = mp.get_current_fig_manager()
    if "window" in platform.system():
        mng.window.state("zoomed")
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
def eval_cv(model, x, y):
    pc = ms.cross_val_score(model, x, y, cv=10,
                            scoring='precision_weighted')
    rc = ms.cross_val_score(model, x, y, cv=10,
                            scoring='recall_weighted')
    f1 = ms.cross_val_score(model, x, y, cv=10,
                            scoring='f1_weighted')
    ac = ms.cross_val_score(model, x, y, cv=10,
                            scoring='accuracy')
    print('{}% {}% {}% {}%'.format(
          round(pc.mean() * 100, 2), round(rc.mean() * 100, 2),
          round(f1.mean() * 100, 2), round(ac.mean() * 100, 2)))
def eval_ac(y,pred_y):
    ac = ((y == pred_y).sum() / pred_y.size)
    print(ac)
def main(argc,argv,envir):
    x, y = read_data("multiple.txt")
    train_x, test_x, train_y, test_y = ms.train_test_split(
        x, y ,test_size=0.25 ,random_state=5
    )
    model = train_model(train_x,train_y)
    l,e,s = x[:,0].min() -1 , x[:,0].max() +1 , 0.005
    b,v,t = x[:,1].min() -1 , x[:,1].max() + 1, 0.005
    grid_x = np.meshgrid(np.arange(l,e,s),
                         np.arange(b,v,t))
    grid_y = pred_model(model,
                        np.c_[grid_x[0].ravel(),grid_x[1].ravel()]).reshape(grid_x[0].shape)
    pred_y = pred_model(model,test_x)
    init_chart()
    draw_point(grid_x,grid_y)
    draw_data(test_x,test_y,pred_y)
    draw_train(train_x,train_y)
    eval_ac(test_y,pred_y)
    eval_cv(model,x,y)
    show_chart()
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))