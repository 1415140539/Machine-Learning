import os
import sys
import platform
import numpy as np
import sklearn.naive_bayes as nb
import sklearn.model_selection as ms
import matplotlib.pyplot as mp
import mpl_toolkits.axes_grid1 as mg
import sklearn.metrics as sm


def read_data(filename):
    x , y = [], []
    with open(filename, "r") as f :
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])
    return np.array(x),np.array(y)

def train_model(x,y):
    model = nb.GaussianNB()
    model.fit(x,y)
    return model
def pred_model(model,x):
    pred_y = model.predict(x)
    return pred_y
def model_ac(model,x,y):
    pc = ms.cross_val_score(model,x,y,cv = 10, scoring = "precision_weighted")
    rc = ms.cross_val_score(model,x,y,cv = 10, scoring = "recall_weighted")
    f1 = ms.cross_val_score(model,x,y,cv = 10 , scoring =  "f1_weighted")
    ac = ms.cross_val_score(model,x,y,cv = 10, scoring="accuracy")
    print("{}% {}% {}% {}%".format(
        round(pc.mean() * 100,2), round(rc.mean() *100 ,2), round(f1.mean() *100 ,2),
        round(ac.mean() * 100 ,2)
    ))
    return True
def eval_ac(y,pred_y):
    res = ((y == pred_y).sum()/pred_y.size)
    print(res)
def eval_cm(y,pred_y):
    cm = sm.confusion_matrix(y,pred_y)
    print(cm)
    return cm
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Naive Bayes Classifier', fontsize=20)
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
def draw_grid(grid_x,grid_y):
    mp.pcolormesh(grid_x[0],grid_x[1],grid_y,cmap = "brg")
    mp.xlim(grid_x[0].min(),grid_x[0].max())
    mp.ylim(grid_x[1].min(),grid_x[1].max())
def draw_train(x,y):
    mp.scatter(x[:,0],x[:,1],marker = "D", c = y, cmap ="RdYlBu")

    return True
def draw_test(x,y,pred_y):
    mp.scatter(x[:,0],x[:,1],c = y, cmap= "RdYlBu")
    mp.scatter(x[:,0],x[:,1],c =pred_y, cmap="RdYlBu")
def init_cm():
    mp.subplot(122)
    mp.title('Confusion Matrix', fontsize=20)
    mp.xlabel('Predicted Class', fontsize=14)
    mp.ylabel('True Class', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, right=True,
                   labelsize=10)
def draw_cm(cm):
    im = mp.imshow(cm, interpolation='nearest', cmap='jet')
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            mp.text(col, row, str(cm[row, col]),
                    color='orangered', fontsize=14,
                    fontstyle='italic', ha='center', va='center')
    dv = mg.make_axes_locatable(mp.gca())
    ca = dv.append_axes('right', '6%', pad='5%')
    cb = mp.colorbar(im, cax=ca)
    cb.set_label('Number Of Samples')
def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
def main(argc,argv,envir):
    x, y = read_data("multiple.txt")
    train_x, test_x, train_y, test_y = ms.train_test_split(
        x, y, test_size = 0.3, random_state= 7
    )
    l,r,s = x[:,0].min()-1,x[:,0].max() +1 ,0.005
    b,t,v = x[:,1].min()-1, x[:,1].max() +1 ,0.005
    grid_x = np.meshgrid(np.arange(l,r,s),np.arange(b,t,v))
    model = train_model(train_x,train_y)
    grid_y = pred_model(model,np.c_[grid_x[0].ravel(),grid_x[1].ravel()]).reshape(grid_x[0].shape)
    pred_y = pred_model(model,test_x)
    model_ac(model,x,y)
    eval_ac(test_y,pred_y)
    cm = eval_cm(test_y,pred_y)
    init_chart()
    draw_grid(grid_x, grid_y)
    draw_train(train_x,train_y)
    draw_test(test_x,test_y,pred_y)
    init_cm()
    draw_cm(cm)
    show_chart()
    return 0
if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))