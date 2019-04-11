import os
import sys
import platform
import numpy as np
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms
import matplotlib.pyplot as mp
import mpl_toolkits.axes_grid1 as mg

def read_data(filename):
    data = []
    encodes = []
    x = []
    with open(filename,"r") as f:
        for line in f.readlines():
            data.append(line[:-1].split(','))
    data = np.array(data).T
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data) -1 :
            x.append(encoder.fit_transform(data[row]))
        else:
            y=encoder.fit_transform(data[row])
        encodes.append(encoder)
    x = np.array(x).T
    return encodes,x,y
def model_ac(test_y,pred_y):
    res = (test_y == pred_y).sum() / pred_y.size
    print(res)
    return res
def train_model(train_x,train_y):
    model = se.RandomForestClassifier(max_depth=7,
                                      n_estimators = 1000,
                                      random_state= 7)
    model.fit(train_x,train_y)
    return model
def pred_model(model,test_x):
    pred_y = model.predict(test_x)
    return pred_y
def model_cv(model,x,y):
    pc = ms.cross_val_score(model,x,y,cv = 2, scoring = "precision_weighted")
    f1 = ms.cross_val_score(model,x,y,cv = 2, scoring="f1_weighted")
    rc = ms.cross_val_score(model,x,y,cv = 2, scoring="recall_weighted")
    ac = ms.cross_val_score(model,x,y,cv = 2, scoring="accuracy")
    print('{}% {}% {}% {}%'.format(
          round(pc.mean() * 100, 2), round(rc.mean() * 100, 2),
          round(f1.mean() * 100, 2), round(ac.mean() * 100, 2)))
def make_data(encoders):
    data = [
        ['high', 'med',  '5more', '4', 'big',   'low',  'unacc'],
        ['high', 'high', '4',     '4', 'med',   'med',  'acc'],
        ['low',  'low',  '2',     '4', 'small', 'high', 'good'],
        ['low',  'med',  '3',     '4', 'med',   'high', 'vgood']]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        if row < len(data) - 1:
            x.append(encoder.transform(data[row]))
        else:
            y = encoder.transform(data[row])
    x = np.array(x).T
    return x, y
def model_cm(y,pred_y):
    cm = sm.confusion_matrix(y,pred_y)
    print(cm)
    return cm
def model_cr(y,pred_y):
    cr = sm.classification_report(y,pred_y)
    print(cr)

def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240/255)
    mp.title("Confusion Matrix",fontsize = 20)
    mp.xlabel("x",fontsize = 10)
    mp.ylabel("y",fontsize = 10)
    mp.tick_params(which = "both", right = True,
                   top =True, labelright = False,
                   labelsize= 10)
    mp.grid(linestyle =":")
def drawl_chart(cm):
    im = mp.imshow(cm,interpolation = "nearest",cmap="RdYlBu")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            mp.text(col,row,str(cm[row,col]),
            color = "black", fontsize = 14,
                    ha = "center",va = "center",
                    fontstyle = "italic")
    dv = mg.make_axes_locatable(mp.gca())
    ca = dv.append_axes("right","6%",pad="5%")
    cb = mp.colorbar(im,cax = ca)
    cb.set_label('Number Of Samples')
    mp.show()
    return None
def main(argc,argv,envir):
    encodes, train_x, train_y = read_data("car.txt")
    model = train_model(train_x,train_y)
    test_x,test_y = make_data(encodes)
    pred_y = pred_model(model,test_x)
    model_cv(model,train_x,train_y)
    model_ac(test_y,pred_y)
    cm = model_cm(test_y,pred_y)
    model_cr(test_y,pred_y)
    print(encodes[-1].inverse_transform(test_y))
    print(encodes[-1].inverse_transform(pred_y))
    init_chart()
    drawl_chart(cm)
if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv,os.environ))