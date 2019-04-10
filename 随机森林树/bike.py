import os
import csv
import sys
import platform
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.model_selection as sm
import matplotlib.pyplot as mp
import sklearn.metrics as ms


def read_data(filename,fb,fe):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        x, y = [], []
        for row in reader:
            x.append(row[fb:fe])
            y.append(row[-1])
        fn = np.array(x[0])
        x = np.array(x[1:], dtype=float)
        y = np.array(y[1:], dtype=float)
        x, y = su.shuffle(x, y, random_state=7)
        return x,y,fn
def train_model(train_x,train_y):
    model = se.RandomForestRegressor(max_depth=4,
                                     n_estimators = 1000,
                                     min_samples_split= 2 )
    model.fit(train_x,train_y)
    return model
def pred_model(model,test_x):
    pred_y = model.predict(test_x)
    return pred_y
def eval_model(y,pred_y):
    mae = ms.mean_absolute_error(y,pred_y)
    mse = ms.mean_squared_error(y,pred_y)
    mbe = ms.median_absolute_error(y,pred_y)
    evs = ms.explained_variance_score(y,pred_y)
    r2 = ms.r2_score(y,pred_y)
    print(mae,"|",mse,"|",mbe,"|",evs,"|",r2)
    return  True
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240/255)
    mp.subplot(211)
    mp.title("Features Importances",fontsize = 24)
    mp.xlabel("features", fontsize = 16)
    mp.ylabel("features", fontsize = 16)
    mp.tick_params(which = "both", top = True, right = True,
                   labelright = True, labelsize = 10)
    mp.grid(linestyle = ":")
def draw_chart(fn,fi):
    fi = fi * 100
    x = np.arange(fi.size)
    sorted_indics = np.flipud(fi.argsort())
    mp.bar(x,fi[sorted_indics],width = 0.8, ec ="limegreen", fc = "orangered", label = "Day")
    mp.xticks(x,fn[sorted_indics])
def draw_chart2(fn,fi):
    mp.subplot(212)
    fi = fi * 100
    x = np.arange(fi.size)
    sorted_indics = np.flipud(fi.argsort())
    mp.bar(x,fi[sorted_indics],width = 0.8, ec ="yellow", fc = "dodgerblue", label = "Hour")
    mp.xticks(x,fn[sorted_indics])
def show():
    mng = mp.get_current_fig_manager()
    if "window" in platform.system():
        mng.window.state("zoomed")
    else:
        mng.resize(*mng.window.maxsize())
    mp.tight_layout()
    mp.show()
def main(argc, argv, envir):
    x, y, fn = read_data("bike_day.csv",2,13)
    h_x,h_y,h_fn = read_data("bike_hour.csv",2,13)
    train_x, test_x, train_y, test_y = sm.train_test_split(
        x,y,test_size = 0.3, random_state=7
    )
    h_train_x,h_test_x, h_train_y, h_test_y = sm.train_test_split(
        h_x, h_y, test_size=0.3, random_state=7
    )
    model = train_model(train_x, train_y)
    model2 = train_model(h_train_x,h_train_y)
    pred_y = pred_model(model,test_x)
    h_pred_y = pred_model(model2,h_test_x)
    eval_model(test_y,pred_y)
    eval_model(h_test_y,h_pred_y)
    fi_day = model.feature_importances_
    fi_hour = model2.feature_importances_
    init_chart()
    draw_chart(fn,fi_day)
    draw_chart2(h_fn,fi_hour)
    show()
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv),sys.argv, os.environ))
