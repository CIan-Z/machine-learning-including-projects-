
import os
os.environ['OMP_NUM_THREADS'] = '4' #OpenMP多任务程序，这里设置为4个线程，防止出现线程冲突


from sklearn.cluster import KMeans                   #聚类的API，采用指定质心分簇
import matplotlib.pyplot as plt                      #绘图
from sklearn.datasets import make_blobs              #默认按照高斯分布生成数据集，需要指定均值，标准差
from sklearn.metrics import calinski_harabasz_score  #评价指标，值越大，聚类效果越好

#1 准备数据集
#(样本数量，特征数量，标签数量，标准差，随机种子）
x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=23)
print(x)
print(y)
#2 绘图
#(横坐标，纵坐标，颜色） x轴：取特征矩阵 x 的第一列（索引0）的所有行
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()
#3 创建 Kmeans对象
estimator = KMeans(n_clusters=4,random_state=23)
#4 训练和预测
y_pred = estimator.fit_predict(x)

#5 绘制预测结果

plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.show()

#6 评价指标
print(f'评分：{calinski_harabasz_score(x,y_pred)}') #越大越好