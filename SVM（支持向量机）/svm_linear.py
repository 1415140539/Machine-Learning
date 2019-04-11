import os
import sys
import platform
import numpy as np
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.metrics as sm
import matplotlib.pyplot as mp


#一个类被另一个类包围，这种形式的数据属于线性不可分割状态，
# 不能使用线性分类器
def read_data(filename):
    x, y = [], []
    with open(filename, "r") as f:
        for line in f.readlines():
            data =[float(substr) for substr in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])

    return np.array(x),np.array(y)

def train_model(train_x,train_y):
    model = svm.SVC(kernel = 'linear')
    model.fit(train_x,train_y)
    return model
def pred_model(model,test_x):
    pred_y = model.predict(test_x)
    return pred_y
def eval_cr(y, pred_y):
    cr = sm.classification_report(y, pred_y)
    print(cr)
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 /255)
    mp.title("SVM Linear",fontsize = 24)
    mp.xlabel("X",fontsize=16)
    mp.ylabel("Y",fontsize = 16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which = "both",right = True,
                   top = True, labelright = True,
                   labelsize = 10)
def draw_grid(grid_x,grid_y):
    mp.pcolormesh(grid_x[0],grid_x[1],grid_y,cmap = "gray")
    mp.xlim(grid_x[0].min(),grid_x[0].max())
    mp.ylim(grid_x[1].min(),grid_x[1].max())
def draw_data(x,y):
    C0, C1 = y==0, y==1
    mp.scatter(x[C0][:, 0], x[C0][:, 1], c='orangered', s=80)
    mp.scatter(x[C1][:, 0], x[C1][:, 1], c='limegreen', s=80)
def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
def main(argc,argv,envir):
    x, y = read_data("binary.txt")
    l,e,s = x[:,0].min()-1, x[:,0].max() + 1, 0.005
    b,t,v = x[:,1].min()-1, x[:,1].max() +1 ,0.005
    train_x, test_x, train_y, test_y = ms.train_test_split(
        x, y, test_size=0.25, random_state=5)
    grid_x = np.meshgrid(np.arange(l,e,s),
                np.arange(b,t,v))
    model = train_model(train_x,train_y)
    grid_y = pred_model(model, np.c_[grid_x[0].ravel(),
                                    grid_x[1].ravel()]).reshape(grid_x[0].shape)

    pred_y = pred_model(model,test_x)
    eval_cr(test_y,pred_y)
    init_chart()
    draw_grid(grid_x,grid_y)
    draw_data(x,y)
    show_chart()
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))