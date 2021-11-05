# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/11/4 15:07
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : get_dataset_from_sarc.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import pandas as pd

# 如果你需要其他的话题，可以打开这个链接https://nlp.cs.princeton.edu/SARC/1.0/main/，下载train-balanced.csv.bz2和test-balanced.csv.bz2解压就可以了
sarc_train_file = r'D:\code\pycharm\cascade_sarc\data\train-balanced.csv'
sarc_test_file = r'D:\code\pycharm\cascade_sarc\data\test-balanced.csv'


df_sarc_train = pd.read_csv(sarc_train_file, header=None, sep='\t', usecols=[0, 1, 3], names=['label', 'text', 'forum'])
df_sarc_test = pd.read_csv(sarc_test_file, header=None, sep='\t', usecols=[0, 1, 3], names=['label', 'text', 'forum'])

# 论坛名称,具体参照数据集中的名称!!
forum_list = ['movies', 'technology']

# 按照论坛名称进行分组
group_train = df_sarc_train.groupby('forum')
group_test = df_sarc_test.groupby('forum')

# 分别保存所需要论坛的label和text
for name in forum_list:
    # 获得论坛训练数据dataframe
    df = group_train.get_group(name)
    # 保存csv文件
    df.to_csv(name + '_train.csv', index=False, sep='\t')

    # 获得论坛测试数据dataframe
    df = group_test.get_group(name)
    # 保存csv文件
    df.to_csv(name + '_test.csv', index=False, sep='\t')
