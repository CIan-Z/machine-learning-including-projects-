import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1.加载数据
data = pd.read_csv('./data/train.csv')
data.info()
print(data.head())
#2.数据预处理
#2.1提取特征和标签
x = data[['Pclass','Sex','Age']]
y = data['Survived']

print(x.head(5))
print(y.head(5))

#2.2 发现Age列有缺失，用该列的平均值作填充
#x['Age'].fillna(x['Age'].mean(), inplace=True)#有警告，但可以用
#x['Age'] = x['Age'].fillna(x['Age'].mean()) #还会有警告，因为是直接修改源数据
#解决方案，copy（）数据之后在修改
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())



#2.3查看处理后的数据
#2.4对sex列进行one hot处理，使之成为0/1区间
x = pd.get_dummies(x,columns=['Sex'])

#2.5划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)

#3  特征工程



#4 模型训练
estimator = DecisionTreeClassifier(max_depth=10)  #max_depth=10是，绘制的决策树，最多10层
estimator.fit(x_train,y_train)
#5模型预测
y_pred = estimator.predict(x_test)
print('预测值为：{}'.format(y_pred))

#6 模型评估
print(f'分类评估报告:\n{classification_report(y_test,y_pred)}')

#7 绘制决策树 图
plt.figure(figsize=(30,20)) #设置图片大小   之后图片是3000×2000像素，是因为30×100（dpi）×20×100（dpi）
#参1：模型对象 参2：是否用颜色填充 参3：绘制的决策树结构，最多10层。
plot_tree(estimator, filled=True, max_depth=10)
plt.savefig('./data/my_titanic.png')
plt.show()