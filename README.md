我的机器学习之路

机会时靠自己争取的。别人能做到相信自己也行。逃离"温室"

我的机器学习


偏差：模型对于不同的训练样本集，预测结果的平均误差。

方差：模型对于不同训练样本集的敏感程度。

噪声：数据集本身的一项属性

通过减少特征数量，可以避免出现过拟合问题，从而避免“维数灾难”。

2019-4-09 5:50->岭回归alpha 正则化直线  fit_intercept 计算直线的截距, max_iter 最大迭代次数 （不同的alpha 对直线得到影响较大）
sl.Ridge(alpha,fit_intercept,max_iter = 1000)

岭回归跟线性回归比较增加了正则化，可以防止过拟合，采用L2惩罚，加上所有参数的平方和

Lasso跟岭回归的比较 是采用了L1惩罚 (绝对值)，加入所有参数的绝对值和

Lasso是另一种数据降维方法，该方法不仅适用于线性情况，也适用于非线性情况。Lasso是基于惩罚方法对样本数据进行变量选择，通过对原本的系数进行压缩，将原本很小的系数直接压缩至0，从而将这部分系数所对应的变量视为非显著性变量，将不显著的变量直接舍弃。
代码见 ridge.py 

2019-4-10 5:50->多项式回归 degree 最高次指数,太大容易过拟合

model = si.make_pipeline(sp.PolynomialFeatures(degree),sl.LinearRegression())

代码见 poly.py

2019-4-10 6：08 -> 决策树跟自适应增强树模型 以及特征的重要性 （波士顿房价的比较）

t_model = st.DecisionTreeRegressor(max_depth = n)
决策树自身的优点

计算简单，易于理解，可解释性强；

比较适合处理有缺失属性的样本；

能够处理不相关的特征；

在相对短的时间内能够对大型数据源做出可行且效果良好的结果。

缺点

容易发生过拟合（随机森林可以很大程度上减少过拟合）；

忽略了数据之间的相关性；

对于那些各类别样本数量不一致的数据，在决策树当中,信息增益的结果偏向于那些具有更多数值的特征（只要是使用了信息增益，都有这个缺点，如RF）。

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

属于判别式模型，有很多正则化模型的方法（L0， L1，L2，etc），而且你不必像在用朴素贝叶斯那样担心你的特征是否相关。与决策树与SVM机相比，你还会得到一个不错的概率解释，你甚至可以轻松地利用新数据来更新模型（使用在线梯度下降算法，online gradient descent）。如果你需要一个概率架构（比如，简单地调节分类阈值，指明不确定性，或者是要获得置信区间），或者你希望以后将更多的训练数据快速整合到模型中去

当特征空间很大时，逻辑回归的性能不是很好；

容易欠拟合，一般准确度不太高

不能很好地处理大量多类特征或变量

只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；

对于非线性特征，需要进行转换；

2019-4-10 22:00 -> 朴素贝斯分类器 以及交叉验证  贝叶斯规则: 贝叶斯规则用于从先验和似然计算后验

朴素贝叶斯分类是一种十分简单的分类算法，叫它朴素贝叶斯分类是因为这种方法的思想真的很朴素，朴素贝叶斯的思想基础是这样的：对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别

朴素贝叶斯模型发源于古典数学理论，有着坚实的数学基础，以及稳定的分类效率。

对小规模的数据表现很好，能个处理多分类任务，适合增量式训练；

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

优点

adaboost是一种有很高精度的分类器。

可以使用各种方法构建子分类器，Adaboost算法提供的是框架。

当使用简单分类器时，计算出的结果是可以理解的，并且弱分类器的构造极其简单。

简单，不用做特征筛选。

不容易发生overfitting。

关于随机森林和GBDT等组合算法，参考这篇文章：机器学习-组合算法总结

缺点：对outlier比较敏感
2019-4-11-10:10 ->交叉验证曲线(随机森林树)

imoprt sklearn.model_selection as ms

验证曲线 train_scores,test_scores = ms.validation_curver(model,x,y,"n_estimators",n_estimators,cv = n) #"分号中的是要测量的属性
学习曲线 train_sizes,train_scores,test_scores = ms.learning_curver(model,x,y,train_size = n, cv =m) 

验证曲线判定过拟合于欠拟合。
验证曲线是非常有用的工具，他可以用来提高模型的性能，原因是他能处理过拟合和欠拟合问题

验证曲线是一种通过定位过拟合于欠拟合等诸多问题的方法，帮助提高模型性能的有效工具。

验证曲线绘制的是准确率与模型参数之间的关系

learning_curve中的train_sizes参数控制产生学习曲线的训练样本的绝对/相对数量，我们设置的train_sizes=np.linspace(0.1, 1.0, 10)，将训练集大小划分为10个相等的区间。learning_curve默认使用分层k折交叉验证计算交叉验证的准确率。

2019-4-11 11:03 -> SVM（支持向量机）

import sklearn.svm as svm

model = svm.SVC(kernel = "")  kernel = "linear" , "poly" + degree  , rbf + C + gamma  + probability

当一个类被另一个类包围，这种形式的数据属于线性不可分割状态，这样的样本根本不适合线性分类器

某一个类型的数据量可能比其他类型多很多，这种条件下训练的分类器就会存在较大的偏差，边界线不能反映出数据的真是特性，因此需要考虑修正样本的比例，或者想办法调和。

degree -- > 多项式的最高次数
gammma --- > 默认为 1/(n_features)
predict_proba(x) 预测结果的分类概率

优点

可以解决高维问题，即大型特征空间；

能够处理非线性特征的相互作用；

无需依赖整个数据；

可以提高泛化能力；

需要对数据提前归一化，很多人使用的时候忽略了这一点，毕竟是基于距离的模型，所以LR也需要归一化



缺点

当观测样本很多时，效率并不是很高；

一个可行的解决办法是模仿随机森林，对数据分解，训练多个模型，然后求平均，时间复杂度降低p倍，分多少份，降多少倍

对非线性问题没有通用解决方案，有时候很难找到一个合适的核函数；

对缺失数据敏感；

2019-4-11 16:00 -> 

缺失值比率 (Missing Values Ratio)   该方法的是基于包含太多缺失值的数据列包含有用信息的可能性较少。因此，可以将数据列缺失值大于某个阈值的列去掉。阈值越高，降维方法更为积极，即降维越少。

低方差滤波 (Low Variance Filter)

随机森林/组合树 (Random Forests)

高相关滤波 (High Correlation Filter) 

高相关滤波认为当两列数据变化趋势相似时，它们包含的信息也显示。这样，使用相似列中的一列就可以满足机器学习模型。对于数值列之间的相似性通过计算相关系数来表示，对于名词类列的相关系数可以通过计算皮尔逊卡方值来表示。相关系数大于某个阈值的两列只保留一列。同样要注意的是：相关系数对范围敏感，所以在计算之前也需要对数据进行归一化处理

随机森林/组合树 (Random Forests) 它在进行特征选择与构建有效的分类器时非常有用。一种常用的降维方法是对目标属性产生许多巨大的树，然后根据对每个属性的统计结果找到信息量最大的特征子集。

主成分分析 (PCA) 
在进行 PCA 变换后会丧失数据的解释性。如果说，数据的解释能力对你的分析来说很重要，那么 PCA 对你来说可能就不适用了。

2019-4-11 16：43 -> K-means 聚类 针对数据样本间距离的计算方法  -- 利用聚类实现矢量量化

import sklearn.cluster as sc

model = sc.KMeans(init = "k-means++", n_clusters = n, init = m)

n_clusters ：要形成的簇数以及要生成的质心数 默认8

n_init ： int，默认值：10 使用不同质心种子运行k-means算法的时间。在惯性方面，最终结果将是n_init连续运行的最佳输出。

init ：初始化方法，默认为'k-means ++'：'k-means ++'：以智能方式选择初始聚类中心进行k均值聚类

2019-4-11 16.14 --> 凝聚层次聚类

import sklearn.cluster as sc

model = sc.AgglomerativeCluster(linkage = "ward")

link ： {“ward”，“complete”，“average”，“single”}

ward最小化被合并的集群的方差。
average使用两组每次观察的平均距离。
complete最大连锁使用两组中所有观测值之间的最大距离。
single使用两组所有观测值之间的最小距离。


Knn:
优点
可用于非线性分类；

训练时间复杂度为O(n)；
缺点
计算量大；

样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；

需要大量的内存

对数据没有假设，准确度高，对outlier不敏感；

xgboost

高准确率高效率高并发，支持自定义损失函数，既可以用来分类又可以用来回归

可以像随机森林一样输出特征重要性，因为速度快，适合作为高维特征选择的一大利器

在目标函数中加入正则项，控制了模型的复杂程度，可以避免过拟合

支持列抽样，也就是随机选择特征，增强了模型的稳定性

对缺失值不敏感，可以学习到包含缺失值的特征的分裂方向
