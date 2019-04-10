我的机器学习之路

机会时靠自己争取的。别人能做到相信自己也行。自律

我的机器学习

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
