import os 
import sys
import numpy as np
import sklearn.linear_model as sl
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import matplotlib.patches as mc
def read_data(filename):
	x, y = [],[]
	with open(filename, "r") as f:
		for line in f.readlines():
			data = [float(substr) for substr in line.split(",")]
			x.append(data[:-1])
			y.append(data[-1])
	return np.array(x),np.array(y)
def train_model(alpha,x,y):
	model = sl.Ridge(alpha,fit_intercept = True, max_iter = 10000)
	#alpha 正规化强度; 必须是积极的浮动。正则化改善了问题的调节并减少了估计的方差
	#fit_intercept是否计算此模型的截距
	#max_iter 共轭梯度求解器的最大迭代次数 默认是1000
	model.fit(x,y)
	return model
def pred_model(model,x):
	pred_y = model.predict(x)
	return pred_y
def eval_model(y,pred_y):
	mae = sm.mean_absolute_error(y,pred_y)
	mse = sm.mean_squared_error(y,pred_y)
	mbe = sm.median_absolute_error(y,pred_y)
	evs = sm.explained_variance_score(y,pred_y)
	r2 = sm.r2_score(y,pred_y)
	print("mae:{},mse:{},mbe:{},evs:{},r2:{}".format(round(mae,2),round(mse,2),
		round(mbe,2),round(evs,2),round(r2,2)))
def init_char():
	mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
	mp.title('Ridge Regression', fontsize=20)
	mp.xlabel("x",fontsize = 14)
	mp.ylabel('y', fontsize=14)
	mp.tick_params(which='both', top=True, right=True,
       labelright=True, labelsize=10)
	mp.grid(linestyle=':')
def draw_chart(x,y,pred_y01,pred_y02):
	mp.plot(x,y,"s",c = "dodgerblue",label = "Training")
	sorted_indes = x.T[0].argsort()
	mp.plot(x.T[0][sorted_indes],pred_y01[sorted_indes],"--",c="lightskyblue",label ="a = 0 Training" )
	mp.plot(x.T[0][sorted_indes],pred_y02[sorted_indes],"--",c = "limegreen",label = "a = 150 Training")
	mp.legend()
def draw_chart_test(x,y,pred_y01,pred_y02):
	mp.plot(x,y,"s",c = "orangered",label = "Testing")
	mp.plot(x, pred_y01, 'o', c='lightskyblue',
            label='Predicted Testing (α=0)')
	mp.plot(x,pred_y02,"s",c = "orangered",label = "Predicted Testing (a = 150)")
	for i ,j, k  in zip(x,y,pred_y01):
		mp.gca().add_patch(mc.Arrow(i, k, 0, j-k, width = 0.8,ec = "none",fc = "pink"))
	for i ,j, k  in zip(x,y,pred_y02):
		mp.gca().add_patch(mc.Arrow(i, k, 0, j-k, width = 0.8,ec = "none",fc = "red"))
def find_suitable_alpha(x,y):
	model = sl.RidgeCV(fit_intercept = True)
	model.fit(x,y)
	return model
def main(argc,argv,envir):
	x, y = read_data('abnormal.txt')
	train_size = int(x.size*0.8)
	train_x = x[:train_size]
	train_y = y[:train_size]
	model_0 = train_model(0,train_x,train_y)
	model_1 = train_model(150,train_x,train_y)
	pred_t_0 = pred_model(model_0,train_x)
	pred_t_1 = pred_model(model_1,train_x)
	test_x = x[train_size:]
	test_y = y[train_size:]
	pred_01 = pred_model(model_0,test_x)
	pred_02 = pred_model(model_1,test_x)
	print("a = 0")
	eval_model(test_y,pred_01)
	print("a = 150")
	eval_model(test_y,pred_02)
	init_char()
	draw_chart(train_x,train_y,pred_t_0,pred_t_1)
	draw_chart_test(test_x,test_y,pred_01,pred_02)
	find_model = find_suitable_alpha(train_x,train_y)
	print(find_model.alpha_)
	mp.show()
	return 0

if __name__ == "__main__":
	sys.exit(main(len(sys.argv),sys.argv,os.environ))
