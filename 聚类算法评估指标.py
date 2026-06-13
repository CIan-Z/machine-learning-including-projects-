#SSE＋肘部法，sc轮廓系数法，CH轮廓系数法

import os
os.environ['OMP_NUM_THREADS'] = '4'


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score


#SSE 随着K值增加SSE值越小
#1 定义函数SSE＋肘部法
def dm01_sse ():
    # 记录每个SSE值
    sse_list = []
    # 生成数据集
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )
    # for循环遍历，获取到每个K值，计算对应的SSE值，并添加到 sse——list列表中
    for k in range(1, 100):
        estimator = KMeans(n_clusters=k, max_iter=100,random_state=23)
        estimator.fit(x)
        sse_value = estimator.inertia_ ## 获取 SSE
        sse_list.append(sse_value)
    # 绘制sse曲线(创建画布，指定画布的尺寸)
    plt.figure(figsize=(20,10))
    #设置标题
    plt.title('sse value')
    # 设置x轴的刻度
    plt.xticks(range(0,100,1))
    # 添加x轴与y轴标签
    plt.xlabel('k')
    plt.ylabel('sse')
    # 绘制网格
    plt.grid()
    #显示图形 （ K值，K值对应的sse值 ）
    plt.plot(range(1,100),sse_list,marker='o')
    plt.show()
    
#2 定义SC
def dm02_sc():
    # 记录每个sc值
    sc_list = []
    # 生成数据集
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )
    # for循环遍历，获取到每个K值，计算对应的sc值，并添加到 sc——list列表中
    for k in range(2, 100): #考虑簇外，至少两个簇
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        estimator.fit(x)
        y_pred = estimator.predict(x)
        sc_value = silhouette_score(x, y_pred)
        sc_list.append(sc_value)

    # 绘制sc曲线(创建画布，指定画布的尺寸)
    plt.figure(figsize=(20, 10))
    # 设置标题
    plt.title('sc value')
    # 设置x轴的刻度
    plt.xticks(range(0, 100, 1))
    # 添加x轴与y轴标签
    plt.xlabel('k')
    plt.ylabel('sc')
    # 绘制网格
    plt.grid()
    # 显示图形 （ K值，K值对应的sc值 ）
    plt.plot(range(2, 100), sc_list, marker='o')
    plt.show()


#3 定义CH
def dm03_ch():
    # 记录每个ch值
    ch_list = []
    # 生成数据集
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )
    # for循环遍历，获取到每个K值，计算对应的ch值，并添加到 ch——list列表中
    for k in range(2, 100): #考虑簇外，至少两个簇
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        estimator.fit(x)
        y_pred = estimator.predict(x)
        ch_value = calinski_harabasz_score(x, y_pred)
        ch_list.append(ch_value)

    # 绘制ch曲线(创建画布，指定画布的尺寸)
    plt.figure(figsize=(20, 10))
    # 设置标题
    plt.title('ch value')
    # 设置x轴的刻度
    plt.xticks(range(0, 100, 1))
    # 添加x轴与y轴标签
    plt.xlabel('k')
    plt.ylabel('ch')
    # 绘制网格
    plt.grid()
    # 显示图形 （ K值，K值对应的ch值 ）
    plt.plot(range(2, 100), ch_list, marker='o')
    plt.show()
if __name__ == '__main__':
    #dm01_sse()
    #dm02_sc()
    dm03_ch()