我的机器学习之路

机会是留给的人的。别人能做到相信自己也行。自律

我的机器学习

2019-4-09 5:50->岭回归alpha 正则化直线  fit_intercept 计算直线的截距, max_iter 最大迭代次数 （不同的alpha 对直线得到影响较大）
sl.Ridge(alpha,fit_intercept,max_iter = 1000)

代码见 ridge.py 

2019-4-10 5:50->多项式回归 degree 最高次指数,太大容易过拟合

model = si.make_pipeline(sp.PolynomialFeatures(degree),sl.LinearRegression())

代码见 poly.py

2019-4-10 6：08 -> 决策树跟自适应增强树模型 以及特征的重要性 （波士顿房价的比较）
t_model = st.DecisionTreeRegressor(max_depth = n)
ada_model = se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth = n),n_estimator = m, random_state = x)
条形统计图的两个条形公用一个x坐标，可以适当的偏移一点距离

