import pandas as pd
from sklearn.model_selection import train_test_split      #切分训练集和测试集
from sklearn.tree import DecisionTreeClassifier           #决策树
from sklearn.ensemble import RandomForestClassifier       #随机森林算法
from sklearn.model_selection import GridSearchCV          #网格搜索

#1.加载数据
df = pd.read_csv('./data/train.csv')
#df.info()

#2.数据预处理
#2.1 抽取特征 标签
x = df[['Pclass','Sex','Age']]
y = df['Survived']

x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())
x = pd.get_dummies(x,columns=['Sex'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)
#3.特征工程


#4.模型训练
#4.1创建决策树对象，演示：单一的决策树
estimator1 = DecisionTreeClassifier()
#模型训练
estimator1.fit(x_train,y_train)
#5.模型预测
y_pred = estimator1.predict(x_test)
print(f'预测值为：{y_pred}')
#6.模型评估
print(f'决策树模型的准确率：{estimator1.score(x_test,y_test)}')
print('-'*23)

#场景2： 随即森林算法（采用默认参数）
#4.2创建随机森林对象，演示：多个决策树（Bagging）效果。
estimator2 = RandomForestClassifier()
#4.2模型训练
estimator2.fit(x_train,y_train)
#4.3预测
y_pred2 = estimator2.predict(x_test)
print(f'预测值为：{y_pred2}')
#4.4 评估
print(f'随机森林模型的准确率为：{estimator2.score(x_test,y_test)}')

print('-'*23)

#场景3：随机森林算法（采用网格搜索）
 #4.1创建随机森林对象
estimator3 = RandomForestClassifier()
estimator3.fit(x_train,y_train) #记得先训练一次

 #4.2 参数准备。'n_estimators' 对应随机森林的“树的数量”'max_depth' 对应每棵树的“最大深度”
 #执行网格搜索时，GridSearchCV 会自动穷举这两个参数的所有组合（5 × 4 = 20 种），对每一种组合用 3 折交叉验证（cv=3）评估模型性能，最终找出效果最好的那一组超参数。
params = {'n_estimators':[30,50,60,90,110],'max_depth':[2,3,5,7]}
 #4.3创建网格搜索对象 结合 交叉验证。
gs_estimator = GridSearchCV(estimator3,param_grid=params,cv=2)
 #4.4模型训练
gs_estimator.fit(x_train,y_train)
 #4.5 预测
y_pred3 = gs_estimator.predict(x_test)
print(f'模型预测准度：{y_pred3}')
 #4.6评估
print(f'随机森林模型准确率为：{estimator3.score(x_test,y_test)}')
 #4.7获取最佳参数
print(f'最佳参数：{gs_estimator.best_params_}')