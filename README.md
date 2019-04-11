我的机器学习之路

机会时靠自己争取的。别人能做到相信自己也行。自律

我的机器学习


偏差：模型对于不同的训练样本集，预测结果的平均误差。

方差：模型对于不同训练样本集的敏感程度。

噪声：数据集本身的一项属性

2019-4-09 5:50->岭回归alpha 正则化直线  fit_intercept 计算直线的截距, max_iter 最大迭代次数 （不同的alpha 对直线得到影响较大）
sl.Ridge(alpha,fit_intercept,max_iter = 1000)
岭回归跟线性回归比较增加了正则化，可以防止过拟合，采用L2惩罚，加上所有参数的平方和
Lasso跟岭回归的比较 是采用了L1惩罚 (绝对值)，加入所有参数的绝对值和
代码见 ridge.py 

2019-4-10 5:50->多项式回归 degree 最高次指数,太大容易过拟合

model = si.make_pipeline(sp.PolynomialFeatures(degree),sl.LinearRegression())

代码见 poly.py

2019-4-10 6：08 -> 决策树跟自适应增强树模型 以及特征的重要性 （波士顿房价的比较）

t_model = st.DecisionTreeRegressor(max_depth = n)

ada_model = se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth = n),n_estimators = m, random_state = x) 

n_estimators 森林树的数量

条形统计图的两个条形公用一个x坐标，可以适当的偏移一点距离

2019-4-10 8：00 -> 随机森林树模型 (时间周期对特征的影响)

model = se.RandomForestRegressor(max_depth = n, n_estimators = m , random_state = x2, min_simples_split = x)

参数的意思见 params.txt

2019-4-10 9：00 -> 逻辑回归分类器
model = sl.LogisticRegressor(solver = "" , C = n) C 惩罚值，对于样本数量少的可以增加惩罚值

max_iter  int，默认值：100 仅适用于newton-cg，sag和lbfgs求解器。求解器收敛的最大迭代次数。

solver: 该类使用'liblinear'库，'newton-cg'，'sag'和'lbfgs'求解器实现正则化逻辑回归。

'newton-cg'，'sag'和'lbfgs'求解器仅支持使用原始公式的L2正则化。'liblinear'求解器支持L1和L2正则化，具有仅针对L2惩罚的双重公式

2019-4-10 22:00 -> 朴素贝斯分类器 以及交叉验证

import sklearn.naive_bayes as sn

model = sn.GaussianNB()

pc = ms.cross_val_score(model, x, y, cv=10,
                            scoring='precision_weighted')
                            
rc = ms.cross_val_score(model, x, y, cv=10,
                            scoring='recall_weighted')
                            
f1 = ms.cross_val_score(model, x, y, cv=10,
                            scoring='f1_weighted') 
                            
ac = ms.cross_val_score(model, x, y, cv=10,
                            scoring='accuracy')  #指标是准确度

交叉验证用于评估模型的预测性能，尤其是训练好的模型在新数据上的表现，可以在一定程度上减小过拟合。
还可以从有限的数据中获取尽可能多的有效信息。

2019-4-11 6：00 -> 混淆矩阵

import sklearn.metrics as sm

sm.confusion_martix(y,pred_y) 

2019-4-11 8:00  - >性能报告

import learn.metrics as sm

sm.classfication_report(y,pred_y)

2019-4-11 9：00 —> 汽车评估（随机森林树）

import sklearn.preprocessing as sp

#处理数据 (测试数据前先用sp.LabelEncoder()对其进行编码，对于数字特征有意义的不用编码)

2019-4-11-10.10 ->交叉验证曲线(随机森林树)

imoprt sklearn.model_selection as ms

验证曲线 train_scores,test_scores = ms.validation_curver(model,x,y,"n_estimators",n_estimators,cv = n) #"分号中的是要测量的属性
学习曲线 train_sizes,train_scores,test_scores = ms.learning_curver(model,x,y,train_size = n, cv =m) 

验证曲线判定过拟合于欠拟合。
验证曲线是非常有用的工具，他可以用来提高模型的性能，原因是他能处理过拟合和欠拟合问题

验证曲线是一种通过定位过拟合于欠拟合等诸多问题的方法，帮助提高模型性能的有效工具。

验证曲线绘制的是准确率与模型参数之间的关系

learning_curve中的train_sizes参数控制产生学习曲线的训练样本的绝对/相对数量，我们设置的train_sizes=np.linspace(0.1, 1.0, 10)，将训练集大小划分为10个相等的区间。learning_curve默认使用分层k折交叉验证计算交叉验证的准确率。

