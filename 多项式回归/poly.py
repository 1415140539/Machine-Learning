import os 
import sys
import platform
import numpy as np
import sklearn.metrics as sm
import sklearn.pipeline as si
import matplotlib.pyplot as mp
import matplotlib.patches as mc
import sklearn.preprocessing as sp
import sklearn.linear_model as sl
import sklearn.model_selection as se


def init_chart():
	mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
	mp.title("PolyNomialFeature",fontsize = 20)
	mp.xlabel("X",fontsize = 16)
	mp.ylabel("Y",fontsize = 16)
	mp.tick_params(which = "both", top = True, right = True,
		labelright = True, labelsize = 10)
	mp.grid(linestyle = ":")
	mp.gcf().autofmt_xdate()
def draw_chart(train_x,test_x,train_y,test_y,pred_train_y,pred_test_y):
	mp.plot(train_x,train_y,"s",c = "limegreen",label = "Training")
	sorted_indics = train_x.T[0].argsort()
	mp.plot(train_x.T[0][sorted_indics],pred_train_y[sorted_indics],"--",c= "orangered",label = "PredTrain")
	mp.plot(test_x,test_y,"o",c = "dodgerblue", label = "Testing")
	mp.plot(test_x,pred_test_y,"s",c = "lightskyblue",label = "PredTest")
	for x, y, p_y in zip(test_x, test_y, pred_test_y):
		mp.gca().add_patch(mc.Arrow(
			x,p_y,0,y-p_y,width = 0.8, ec = "none", fc = "pink"))
	mp.legend()
	mp.show()
def read_data(filename):
	x, y = [], []
	with open(filename,"r") as f:
		for line in f.readlines():
			data = [float(substr) for substr in line.split(",")]
			x.append(data[:-1])
			y.append(data[-1])
	return np.array(x),np.array(y)
def train_model(degree,train_x,train_y):
	model = si.make_pipeline(sp.PolynomialFeatures(degree),
		sl.LinearRegression())
	model.fit(train_x,train_y)
	return model
def pred_model(model,test_x):
	pred_y = model.predict(test_x)
	return pred_y
def eval_model(y,pred_y):
	mae = sm.mean_absolute_error(y,pred_y)
	mse = sm.mean_squared_error(y,pred_y)
	mbe = sm.median_absolute_error(y,pred_y)
	evs = sm.explained_variance_score(y,pred_y)
	r2 = sm.r2_score(y,pred_y)
	print(mae, mse, mbe, evs, r2)
def main(argc, argv, envir):
	x, y = read_data("single.txt")
	train_x, test_x, train_y, test_y = se.train_test_split(x,y,test_size = 0.3 ,random_state = 7)
	model = train_model(9,train_x, train_y)
	pred_train_y = pred_model(model,train_x)
	pred_y = pred_model(model,test_x)
	eval_model(test_y,pred_y)
	init_chart()
	draw_chart(train_x, test_x, train_y, test_y, pred_train_y, pred_y)

	return 0

if __name__ == "__main__":
	sys.exit(main(len(sys.argv),sys.argv,os.environ))