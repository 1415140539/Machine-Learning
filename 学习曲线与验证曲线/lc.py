import os
import sys
import platform
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import matplotlib.pyplot as mp


def read_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            # 去掉回车符
            data.append(line[:-1].split(","))
    data = np.array(data).T
    x = []
    encoders = []
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data)-1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return x, y, encoders
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240/255)
    mp.title("Learning Curver",fontsize = 24)
    mp.xlabel("X",fontsize = 16)
    mp.ylabel("Y",fontsize = 16)
    mp.tick_params(which = "both", right = True, top = True,
                   labelright = True,labelsize = 10)
    mp.grid(linestyle= ":")
def draw_chart(train_sizes,test_score):
    mp.plot(train_sizes,
            test_score.mean(axis=1) * 100,
            'o-', c='dodgerblue', label='Test Score')
    mp.legend()
    return 0
def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
def train_model():
    model = se.RandomForestClassifier(max_depth=4,
                                      n_estimators=400,
                                      random_state= 7)
    return model
def pred_model(model,test_x):
    pred_y = model.predict(test_x)
    return pred_y
def eval_lc(model,x,y,train_sizes):
    train_sizes, train_scores, test_scores = ms.learning_curve(
        model, x, y, train_sizes=train_sizes, cv=5)
    print(train_scores)
    print(test_scores)
    return train_sizes, train_scores, test_scores

def main(argc,argv,envir):
    x, y, encoders = read_data("car.txt")
    model = train_model()
    train_sizes = np.linspace(100, 1000, 10).astype(int)
    train_sizes, train_scores, test_scores = eval_lc(
        model, x, y, train_sizes)
    init_chart()
    draw_chart(train_sizes,test_scores)
    show_chart()
    return 0
if __name__ =="__main__":
    sys.exit(main(len(sys.argv),sys.argv, os.environ))