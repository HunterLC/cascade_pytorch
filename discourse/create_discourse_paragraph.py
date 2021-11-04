# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/18 10:22
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : create_discourse_paragraph.py
@Version : 1.0.0 
@Desc    : 针对每个主题遍历 SARC 1.0 的平衡化训练集，使用 <END> 将同一主题下的所有帖子拼接起来，便于后续doc2vec完成主题嵌入
@LastTime: 
"""

import csv
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# 读取SARC 1.0平衡的训练集
input_data = np.asarray(pd.read_csv("../data/train-balanced.csv", header=None, sep='\t'))
# 统计 主题/子论坛 的数量和具体名字
set_topics = set(input_data[:, 3])
print(set_topics)
print(len(set_topics))

# 生成输出文件夹
if not os.path.exists("./output"):
    os.mkdir("./output")

# 定义输出
output_file = open("./output/train_balanced_topics.csv", 'w')
max_len = 0
wr = csv.writer(output_file, quoting=csv.QUOTE_ALL)

# # 统计信息
# count = 0
# for i in range(len(input_data)):
#     if str(input_data[i][1]) != 'nan' and len(input_data[i][1].split()) > 500:
#         count += 1
#     if str(input_data[i][1]) != 'nan' and max_len < len(input_data[i][1].split()):
#         max_len = len(input_data[i][1].split())

# 针对每个主题遍历训练集，使用 <END> 将同一主题下的所有帖子拼接起来
for ix, topic in tqdm(enumerate(set_topics)):
    ls = []

    # 把待检测的目标语句拿出来
    comments = input_data[input_data[:, 3] == topic, 1]
    comments = [x for x in comments if str(x) != 'nan']

    # 使用 <END> 将同一主题下的所有帖子拼接起来
    comment = " <END> ".join(comments)

    # 输出行，格式“主题名 所有评论内容”
    ls.append(topic)
    ls.append(comment)
    wr.writerow(ls)
