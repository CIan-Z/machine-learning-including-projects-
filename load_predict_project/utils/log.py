# -*- coding: utf-8 -*-
import logging
import os
import datetime

import pandas as pd


# from datetime import datetime


class Logger(object):     #面向对象编程
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, root_path, log_name, level='info', fmt='%(asctime)s - %(levelname)s: %(message)s'):
        # 指定日志保存的路径
        self.root_path = root_path

        # 初始logger名称和格式
        self.log_name = log_name

        # 初始格式
        self.fmt = fmt

        # 先声明一个 Logger 对象
        self.logger = logging.getLogger(log_name)

        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))

    def get_logger(self):
        # 指定对应的 Handler 为 FileHandler 对象， 这个可适用于多线程情况
        path = os.path.join(self.root_path, 'log')
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, self.log_name + '.log')
        rotate_handler = logging.FileHandler(file_name, encoding="utf-8", mode="a")

        # Handler 对象 rotate_handler 的输出格式
        formatter = logging.Formatter(self.fmt)
        rotate_handler.setFormatter(formatter)

        # 将rotate_handler添加到Logger
        self.logger.addHandler(rotate_handler)

        return self.logger


# 测试日志功能.
if __name__ == '__main__':
    # 1. 创建日志对象.
    # 参1: 日志文件的父目录.  参2: 日志文件名
    logger = Logger('../', 'hg_test').get_logger()

    # 2. 往日志文件中写数据.
    logger.info('我是普通的日志信息')   # 这个会被写到 hg.test日志文件中.
    logger.info('我刚增添了日志信息')   # 日志默认是追加, 不是覆盖.

    # 3. 演示下错误信息, 写入日志
    try:
        logger.info('开始计算了...')
        print(10 / 0)
    except Exception as e:
        logger.error(f'计算出错了, 原因是: {e}')
    else:
        logger.info('计算成功!')
    finally:
        logger.info('计算结束!')

    # 4. 生成 train_年月日.log 日志文件.
    print(datetime.datetime.now())      # 获取当前时间.
    print(pd.to_datetime(datetime.datetime.now()).strftime('%Y%m%d%H%M%S'))

    new_name = 'train_' + pd.to_datetime(datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + '.log'
    print(new_name)
