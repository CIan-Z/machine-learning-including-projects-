# -*- coding: utf-8 -*-
# 实现流程：
# 1.导包、配置绘图字体
import os
import pandas as pd
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


# 2.定义电力负荷预测类（PowerLoadPredict），配置日志，获取数据源、历史数据转为字典（避免频繁操作dataframe，提高效率）
class PowerLoadPredict(object):
    def __init__(self, file_path):
        # 配置日志
        # 日志文件名
        logfile_name = 'predict_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger(root_path='../', log_name=logfile_name).get_logger()
        # 获取数据源
        self.data_source = data_preprocessing(file_path)
        # 历史数据转为字典（避免频繁操作dataframe，提高效率）
        self.time_load_dict_all = self.data_source.set_index('time')['power_load'].to_dict()


def feature_extract(time, time_load_dict):
    """
    根据预测时间,历史负荷, 加工出特征数据
    :param time: 预测时间
    :param time_load_dict: 预测时间以前的历史负荷
    :return:
    """
    feature_cols = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06', 'hour_07', 'hour_08',
                    'hour_09',
                    'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                    'hour_19',
                    'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_01', 'month_02', 'month_03', 'month_04',
                    'month_05', 'month_06',
                    'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12', '前1小时',
                    '前2小时',
                    '前3小时', 'yesterday_load']
    # 小时特征
    pred_hour = time[11:13]
    hour_list = []
    # 小时转为编码
    for i in range(24):
        if pred_hour == feature_cols[i][5:7]: #意思是只留‘hour_00’中的00（预测的时间吻合就为1）
            hour_list.append(1)
        else:
            hour_list.append(0)
    # 月份特征
    pred_month = time[5:7]
    month_list = []
    # 月份转为编码
    for i in range(24, 36):#到35结束
        if pred_month == feature_cols[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)

    # 前1小时时间
    last_1h_time = (pd.to_datetime(time) - pd.to_timedelta('1h')).strftime('%Y-%m-%d %H:%M:%S')
    # 前1小时负荷
    last_1h_load = time_load_dict.get(last_1h_time, 500)#如果值不存在，用500填充

    # 前2小时负荷
    last_2h_time = (pd.to_datetime(time) - pd.to_timedelta('2h')).strftime('%Y-%m-%d %H:%M:%S')
    # 前2小时负荷
    last_2h_load = time_load_dict.get(last_2h_time, 500)

    # 前3小时负荷
    last_3h_time = (pd.to_datetime(time) - pd.to_timedelta('3h')).strftime('%Y-%m-%d %H:%M:%S')
    # 前3小时负荷
    last_3h_load = time_load_dict.get(last_3h_time, 500)
    # 昨日同时刻负荷
    last_1d_time = (pd.to_datetime(time) - pd.to_timedelta('1d')).strftime('%Y-%m-%d %H:%M:%S')
    last_1d_load = time_load_dict.get(last_1d_time, 500)

    feature_list = [hour_list + month_list + [last_1h_load, last_2h_load, last_3h_load, last_1d_load]]
    feature_df = pd.DataFrame(feature_list, columns=feature_cols)
    return feature_df

def result_ana(data):
    """
    结果分析, 绘制预测时间-真实负荷折线图,  预测时间-预测负荷折线图
    :param data: 预测结果 ['预测时间', '真实负荷', '预测负荷']
    :return:
    """
    fig = plt.figure(figsize=(40, 20))
    ax  = fig.add_subplot()
    ax.plot(data['预测时间'], data['真实负荷'], label='真实负荷')
    ax.plot(data['预测时间'], data['预测负荷'], label='预测负荷')
    ax.set_title('真实负荷与预测负荷对比结果')
    ax.set_xlabel('预测时间')
    ax.set_ylabel('负荷')
    # 横坐标时间若不处理太过密集，这里调大时间展示的间隔
    ax.xaxis.set_major_locator(mick.MultipleLocator(50))
    # 时间展示时旋转45度
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('../data/fig/真实负荷与预测负荷对比结果图.png')



if __name__ == '__main__':

    pred_obj = PowerLoadPredict('../data/test.csv')
    # 3.加载模型
    model = joblib.load('../model/power_load_model.pkl')
    # 4.模型预测(重点）
    #     4.1 确定要预测的时间段（2015-08-01 00:00:00及以后的时间）
    pred_times = [t for t in pred_obj.data_source['time'] if t >= '2015-08-01 00:00:00']
    evaluate_list = []
    for pred_time in pred_times:
        print(f"开始预测{pred_time}负荷")
        #     4.2 为了模拟实际场景的预测，把要预测的时间以及以后的负荷都掩盖掉，因此新建一个数据字典，只保存预测时间以前的数据字典
        time_load_dict_masked = {k: v for k, v in pred_obj.time_load_dict_all.items() if k < pred_time}
        #     4.3 预测负荷
        #         4.3.1 解析特征（定义解析特征方法）
        processed_data = feature_extract(pred_time, time_load_dict_masked)
        #         4.3.2 利用加载的模型预测
        y_pred = model.predict(processed_data) #相当于用给的数据，根据当天之前的数据，把当天之后的数据给预测出来了
        print(f'模型预测值:{y_pred}')
        #     4.4 保存预测时间对应的真实负荷
        true_value = pred_obj.time_load_dict_all.get(pred_time)
        #     4.5 结果保存到evaluate_list，三个元素分别是预测时间、真实负荷、预测负荷，方便后续进行预测结果评价
        evaluate_list.append([pred_time, true_value, y_pred[0]])
    #     4.6 循环结束后，evaluate_list转为DataFrame
    evaluate_df = pd.DataFrame(evaluate_list, columns=['预测时间', '真实负荷', '预测负荷'])
    # 5.预测结果评价
    #     5.1 计算预测结果与真实结果的MAE
    print(f"预测结果与真实结果之间的平均绝对误差:{mean_absolute_error(evaluate_df['真实负荷'], evaluate_df['预测负荷'])}")
    #     5.2 绘制折线图（预测时间-真实负荷折线图，预测时间-预测负荷折线图），查看预测效果
    result_ana(evaluate_df)
