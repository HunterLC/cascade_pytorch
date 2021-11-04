# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/24 15:48
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : generate_user_personality.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import pickle
import numpy as np
import pandas as pd
import re
import csv
import torch
from torch.utils.data import DataLoader
from user.personality.TextCNN import TextCNN, PersonalityDataset
from tqdm import tqdm


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels_test(file=None, partial=True):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    if partial:
        x = pickle.load(open(r"D:\code\pycharm\cascade_sarc\model\mainbalancedpickle.p", "rb"))
        revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
        users = np.asarray([revs[i]['author'] for i in range(len(revs))])
        paragraphs = np.asarray([revs[i]['text'] for i in range(len(revs))])
    else:
        test_data = np.asarray(pd.read_csv(file, header=None, sep='\t'))
        users = test_data[:, 2]
        paragraphs = test_data[:, 1]
    # Labels = []
    # for i in range(len(test_data)):
    #     Labels.append([int(test_data[i][j]) for j in range(1,6,1)])
    paragraphs = [(x.strip())[0:1000] for x in paragraphs if str(x) != 'nan']
    paragraphs = [clean_str(x) for x in paragraphs]
    paragraphs = np.asarray(paragraphs)
    # Labels = np.asarray(Labels)
    return [users, paragraphs]


def read_data():
    x = pickle.load(open(r"D:\code\pycharm\cascade_sarc\user\personality\personalitypickle.p", "rb"))
    revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    return W, word_idx_map, vocab


def get_para(x_test, mp_vec, mp_count):
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    test_set = PersonalityDataset(x_test)
    # Construct data loaders.
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)  # only shuffle the training data

    with torch.no_grad():
        # Initialize a model, and put it on the device specified.
        model = TextCNN(sequence_length=1000, num_classes=5, vocab_size=vocab, word2vec_W=W,
                        embedding_dim=300, kernel_sizes=[3, 4, 5], num_kernel=128).to(device)
        # 加载保存的状态参数
        # model.load_state_dict(torch.load(r"D:\code\pycharm\cascade_sarc\user\personality\output\best_steps_94.pt", map_location='cpu'))
        model.load_state_dict(
            torch.load(r"D:\code\pycharm\cascade_sarc\user\personality\output\best_steps_23.pt"))
        model.device = device

        user_ind = 0
        for batch in tqdm(test_loader):
            # A batch consists of data and corresponding labels.
            data = batch
            fc1, outputs = model(data.to(device))
            # 把tensor转到cpu上，并转成numpy
            fc1 = fc1.cpu().numpy()
            # print(fc1)
            # break
            # print(outputs.shape)
            for i in range(len(batch)):
                # 数据集users中每个元组所输出的个性特征
                mp_vec[users[user_ind]] += fc1[i]
                # 统计每名用户出现的次数，方便后续做平均
                mp_count[users[user_ind]] += 1
                user_ind += 1
            # _, predicted = torch.max(outputs, 1)
            # print(predicted)
    return mp_vec, mp_count

print("loading data...", )
# 加载数据集，不能全部加载，否则内存爆炸
users, x_raw = load_data_and_labels_test(file=r"D:\code\pycharm\cascade_sarc\data\train-balanced.csv", partial=True)
W, word_idx_map, vocab = read_data()

print("data loaded!")

# 将文本分词后转换为数字索引，便于后续嵌入层查找词向量
x = []
for i in range(len(x_raw)):
    lst = []
    try:
        for word in x_raw[i].split():
            if word in word_idx_map:
                lst.append(word_idx_map[word])
            else:
                lst.append(0)
    except AttributeError:
        lst.append(0)
    x.append(lst)

# 控制统一的输入为1000长度，不够长的补零
for i in range(len(x)):
    if (len(x[i]) < 1000):
        x[i] = np.append(x[i], np.zeros(1000 - len(x[i])))
    elif (len(x[i]) > 1000):
        x[i] = x[i][0:1000]

# 为我们使用的数据集生成对应的用户个性嵌入特征
x_test = np.asarray(x)

# 获取不重复的用户名
user_set = set(users)

mp_vec = {}
mp_count = {}
for user in user_set:
    mp_vec[user] = [0] * 100
    mp_count[user] = 0

print(len(user_set))

# 获得用户嵌入结果
mp_vec, mp_count = get_para(x_test, mp_vec, mp_count)

# 保存【用户名，用户特性嵌入特征】
res = []
for user in user_set:
    # print(user)
    ls = []
    # 按照算法思路，每个用户都应该出现
    if mp_count[user] == 0:
        print("Error")
        exit()

    for i in range(100):
        # 对一个用户特征的每一位分别平均，那为啥不整体平均呢
        mp_vec[user][i] /= mp_count[user]

    # print(user, mp_vec[user])
    ls.append(user)
    ls.append(mp_vec[user])
    res.append(ls)

# 写入csv文件中
with open("./output/user_personality.csv", "w", newline='') as output:
    writer = csv.writer(output)
    for val in res:
        writer.writerow(val)
