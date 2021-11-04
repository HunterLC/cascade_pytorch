# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/22 16:22
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : generate_user_stylometric.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import pandas as pd
import numpy as np
import csv
import gensim
import os

doc2vec = gensim.models.Doc2Vec.load('./output/user_stylometric')
data = np.asarray(pd.read_csv('./output/train_balanced_user.csv', header=None))
DIM = 300

directory = "./output"
if not os.path.exists(directory):
    os.makedirs(directory)
file = open(directory + "/user_stylometric.csv", 'w', newline='')
wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar=' ')

# Inferring paragraphVec vectors for each user
vectors = np.asarray([doc2vec.infer_vector(data[i][1]) for i in range(data.shape[0])])

users = data[:, 0]
for i in range(len(users)):
    ls = []
    ls.append(users[i])
    v = [0] * 100
    for j in range(len(vectors[i])):
        v[j] = vectors[i][j]
    ls.append(v)
    wr.writerow(ls)
