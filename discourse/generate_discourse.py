# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/18 16:01
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : generate_discourse.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import csv
import gensim
import numpy as np
import os

input_model_path = "./output/discourse_model"
output_folder = "./output"

print('loading topics model...')
model = gensim.models.Doc2Vec.load(input_model_path)
print('topics model loaded')

# 保存最终的主题嵌入表示
topic_embeddings = []
# 获得训练模型的嵌入列表
topic_ids = model.docvecs.offset2doctag
topic_embeddings_size = 100

for ix in topic_ids:
    try:
        lst = model.docvecs[ix]
    except TypeError:
        lst = np.random.normal(size=topic_embeddings_size)
    topic_embeddings.append(lst)

# 针对模型中未出现的主题，随机初始化嵌入表示
topic_ids = [0] + topic_ids
unknown_vector = np.random.normal(size=(1, topic_embeddings_size))

# 将随机初始化的主题嵌入与之前的进行拼接
topic_embeddings = np.concatenate((unknown_vector, topic_embeddings), axis=0)
topic_embeddings = topic_embeddings.astype(dtype='float32')
print(topic_embeddings)
# print(topic_embeddings[1])
# print(type(topic_embeddings[1]))
print("len of topic embeddings: ", len(topic_embeddings))

# 主题嵌入表示输出
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
with open(output_folder + "/discourse_embeddings.csv", "w", newline='') as fp:
    csv_writer = csv.writer(fp, quoting=csv.QUOTE_NONE, escapechar=' ')
    for i in range(len(topic_embeddings)):
        # 输出嵌入表示，格式"主题名，嵌入向量"
        # print([topic_ids[i], topic_embeddings[i]])
        csv_writer.writerow([topic_ids[i], str(','.join(list(map(str, topic_embeddings[i].tolist()))))])
