import os 
import sys
import platform
import numpy as np 
import pandas as pd
import sklearn.tree as st
import sklearn.utils as su
import sklearn.metrics as sm
import sklearn.datasets as sd
import sklearn.ensemble as se
import matplotlib.pyplot as mp 
import sklearn.model_selection as ml


def get_data():
	hosing = sd.load_boston()
	features = hosing.feature_names
	x, y = su.shuffle(hosing.data, hosing.target, random_state = 7)
	print(x.shape,y.shape)
	return x, y , features
def tree_train_model(train_x, train_y):
	model = st.DecisionTreeRegressor(max_depth = 4) #设置深度
	model.fit(train_x,train_y)
	return model
def init_chart():
	mp.gcf().set_facecolor(np.ones(3) * 240/255)
	mp.subplot(211)
	mp.title("Tree Model", fontsize = 26)
	mp.xlabel("feature_namse", fontsize= 16)
	mp.ylabel("importance",fontsize=16)
	mp.tick_params(which = "both",top = True,
					right = True, labelright = True, labelsize = 10)
	mp.gcf().autofmt_xdate()
	mp.grid(linestyle = ":")
def draw_t_chart(fn,t_fi):
	t_fi = 100 * t_fi
	x = np.arange(len(fn))
	sorted_indics = np.flipud(t_fi.argsort())
	mp.bar(x,t_fi[sorted_indics],width = 0.8,ec = "limegreen", fc = "orangered", label = "Tree Model")
	mp.xticks(x,fn[sorted_indics])
	mp.legend()

def draw_ada_chart(fn,ada_fi):
	ada_fi = 100 * ada_fi
	mp.subplot(212)
	x = np.arange(len(fn))
	sorted_indics = np.flipud(ada_fi.argsort())
	mp.bar(x,ada_fi[sorted_indics],width = 0.8,ec = "yellow", fc = "dodgerblue", label = "AdaTree Model")
	mp.xticks(x,fn[sorted_indics])
	mp.legend()
def show():
	mng = mp.get_current_fig_manager()
	if "windows" in platform.system():
		mng.window.state("zoomed")
	else:
		mng.resize(*mng.window.maxsize())
	mp.tight_layout()
	mp.show()
def adatree_train_model(train_x,train_y):
	model = se.AdaBoostRegressor(
		st.DecisionTreeRegressor(max_depth = 4),
		n_estimators = 400 ,random_state = 7)
	model.fit(train_x,train_y)
	return model
def pred_model(model,test_x):
	pred_y = model.predict(test_x)
	return pred_y
def eval_model(y,pred_y):
	#评估模型
	mae = sm.mean_absolute_error(y,pred_y) 
	mse = sm.mean_squared_error(y,pred_y)
	mbe = sm.median_absolute_error(y,pred_y)
	evs = sm.explained_variance_score(y,pred_y)
	r2 = sm.r2_score(y,pred_y)
	res = "mae:{} mse:{} mbe:{} evs:{} r2:{}".format(round(mae,2),round(mse,2),
		round(mbe,2),round(evs,2),round(r2,2))
	return res
def main(argv, argc, envir):
	x, y, fn = get_data()
	#生成训练集和 测试集
	train_x, test_x, train_y, test_y = ml.train_test_split(x,
		y, test_size = 0.3, random_state = 7)
	t_model = tree_train_model(train_x,train_y)
	ada_model = adatree_train_model(train_x,train_y)
	pred_y_ada = pred_model(ada_model,test_x)
	pred_y = pred_model(t_model, test_x)
	result = eval_model(test_y,pred_y)
	print(result)
	result = eval_model(test_y,pred_y_ada)
	print(result)
	print("t_model")
	print(t_model.feature_importances_)
	t_fi = t_model.feature_importances_
	print("ada_model")
	print(ada_model.feature_importances_)
	adat_fi = ada_model.feature_importances_
	init_chart()
	draw_t_chart(fn, t_fi)
	draw_ada_chart(fn,adat_fi)
	show()
	return 0

if __name__ == "__main__":
	sys.exit(main(len(sys.argv),sys.argv,os.environ))