# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/11/2 11:25
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : merge_user_embedding.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import pandas as pd
import numpy as np
import csv

doc_data = np.asarray(pd.read_csv('./stylometric/output/user_stylometric.csv', header=None))
per_data = np.asarray(pd.read_csv('./personality/output/user_personality.csv', header=None))
csvfile = open("./user_view_vectors.csv", 'w', newline='')
wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

x = [doc_data[i][1:] for i in range(len(doc_data))]
for i in range(len(doc_data)):
    x[i][0] = x[i][0][1:]
    x[i][len(x[i]) - 1] = x[i][len(x[i]) - 1][:-1]

# numpy类型的字符串存在换行和若干空格，所以需要split
y = [per_data[i][1].replace('[', '').replace(']', '').lstrip().rstrip().split() for i in range(len(per_data))]
# for i in range(len(per_data)):
#     # 去除每一行数据的第一个和最后一个数据中存在的多余括号[]
#     y[i][0] = y[i][0][1:]
#     y[i][len(y[i]) - 1] = y[i][len(y[i]) - 1][:-1]

x = np.asarray(x)
y = np.asarray(y)

map_dv = {}
for i in range(len(doc_data)):
    map_dv[doc_data[i][0]] = x[i]

map_pv = {}
for i in range(len(per_data)):
    map_pv[per_data[i][0]] = y[i]
# print(map_dv["0"])
users = per_data[:, 0]
for user in users:
    ls = []
    ls.append(user)
    # 默认两个向量都是100维
    try:
        ls.append(float(len(map_dv[user])))
    except KeyError:
        ls.append(float(100))
    ls.append(float(len(map_pv[user])))
    for j in range(100):
        try:
            ls.append(float(map_dv[user][j]))
        except KeyError:
            ls.append(float(0))
    for j in range(len(map_pv[user])):
        try:
          ls.append(float(map_pv[user][j]))
        except ValueError:
            print(user,j)
            exit()
    wr.writerow(ls)
