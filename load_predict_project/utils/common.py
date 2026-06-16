# 导包
import numpy as np
import pandas as pd


# 该工具类的目的是: 对数据做预处理 -> 时间格式化, 按照时间升序排列, 且对数据去重.
# 数据集在 data目录下的 train.csv文件中 -> 拆分训练集 和 测试集.
# 测试集在 data目录下的 test.csv文件中 -> 模拟项目上线后, 真实的测试集.

# 定义函数 data_preprocessing(), 对数据做预处理操作.
def data_preprocessing(file_path):
    # 1. 加载数据集.
    #data = pd.read_csv('../data/train.csv')
    data = pd.read_csv(file_path)
    data.info()

    # 2. 时间格式化, 转为: '%Y-%m-%d %H:%M:%S'
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 3. 按照时间升序排列.
    data.sort_values('time', ascending=True, inplace=True)

    # 4. 去重.
    data.drop_duplicates(inplace=True)

    # 5. 打印和返回.
    #print(data)
    return data

if __name__ == '__main__':
    data_preprocessing()