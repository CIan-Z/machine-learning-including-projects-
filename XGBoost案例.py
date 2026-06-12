import joblib                                           #保存和加载模型
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter                         #统计数据
from sklearn.model_selection import train_test_split, GridSearchCV  # 训练集和测试集的划分
from sklearn.metrics import classification_report, accuracy_score  # 模型分类评估报告
from sklearn.model_selection import StratifiedKFold     #分层K折交叉验证，类似于网格搜索时CV=折数
from sklearn.utils import class_weight                  #计算样本权重



#1.定义函数，对 红酒品质分类源数据→拆分成训练集和测试集，存储到csv文件中

def dm01_data_split():
    #1.1加载数据集
    df = pd.read_csv('./data/红酒品质分类.csv')
    df.info()
    #1.2抽取特征数据和标签数据
    x = df.iloc[:,:-1]       #选取从第一列到倒数第二列
    y = df.iloc[:,-1] - 3    #最后一列是标签，默认范围是：3~8 → 0~5
    # 1.3 查看数据
    # print(x[:5])
    # print(y[:5])
    # print(f'查看标签分布是否均衡：{Counter(y)}')



    #1.4 切分训练集和测试集 从传入的原始数据集中切分出来的
    #参1：特征数据 参2：标签数据 参3：测试的比例 参4：随机种子 参5：确保训练/测试集中各类别比例与原始数据集一致
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)

    #1.5 把上述的数据集和标签  ，测试集数据集和标签分别拼接到一起，最后写到文件中
    #print(pd.concat([x_train, y_train], axis=1))
    pd.concat([x_train,y_train],axis=1).to_csv('./data/红酒品质分类_train.csv',index=False)
    pd.concat([x_test,y_test],axis=1).to_csv('./data/红酒品质分类_test.csv',index=False)

#2. 定义函数 训练模型和保存

def dm02_train_model():
    # 2.1 读取训练集测试集.
    train_data = pd.read_csv('./data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./data/红酒品质分类_test.csv')

    #2.2 提取训练集和测试集的 特征数据和标签数据
    x_train = train_data.iloc[:,:-1]  #除了最后一列，都是特征
    y_train = train_data.iloc[:,-1]   #最后一列是标签
    x_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    #2.3 创建模型对象
    estimator = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=100,         #树的数量
        learning_rate=0.1,
        random_state=23,
        objective='multi:softmax'  #softmax多分类模型
    )  #如果用的是from XGB，则可以直接用XGBClassifier()，而不用xgb.XGBClassifier()

    #加入平衡权重，因为数据集样本不均衡                 参1：平衡权重  参2：参考标签来平衡
    class_weight.compute_sample_weight('balanced',y_train)

    #2.4 模型训练与评估

    estimator.fit(x_train,y_train)
    print(f'准确率：{estimator.score(x_test,y_test)}')

    #2.5 保存模型

    joblib.dump(estimator,'./model/红酒品质分类.pkl')  #后缀名也可以写 .pth，都是pickle文件格式
    print('模型保存成功！')


#3 定义函数，测试模型
def dm03_train_model():
    # 3.1 读取训练集测试集.
    train_data = pd.read_csv('./data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./data/红酒品质分类_test.csv')

    # 3.2 提取训练集和测试集的 特征数据和标签数据
    x_train = train_data.iloc[:, :-1]  # 除了最后一列，都是特征
    y_train = train_data.iloc[:, -1]  # 最后一列是标签
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    #3.3 加载模型
    estimator = joblib.load('./model/红酒品质分类.pkl')
    #3.4创建网格搜索，交叉验证，找模型最优参数
    #定义变量
    param_dict = {'max_depth':[2,3,5,6,7],'n_estimators':[30,60,100,130],'learning_rate':[0.4,0.5,1,1.4]}
    #创建分层采样对象 参1：折数 参2：是否打乱（数据) 参3：随机种子
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
    #创建网格搜索交叉验证对象
    gs_estimator = GridSearchCV(estimator,param_dict,cv=skf)
    #模型训练
    gs_estimator.fit(x_train,y_train)
    #模型预测
    y_pred = gs_estimator.predict(x_test)
    print(f'预测值：{y_pred}')
    #打印模型评估系数
    print(f'最优参数组合：{gs_estimator.best_params_}')
    print(f'最优评分：{gs_estimator.best_score_}')
    print(f'最终准确率：{accuracy_score(y_test,y_pred)}')

#4.测试
if __name__ == '__main__':  #这个函数只在直接执行脚本时运行，在其他脚本里不会运行
      #dm01_data_split()
      #dm02_train_model()
      dm03_train_model()