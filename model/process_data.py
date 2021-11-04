# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/19 9:31
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : process_data.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

from __future__ import print_function

import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import json

# noinspection PyCompatibility
from builtins import range

COMMENTS_FILE = "../data/comments.json"
TRAIN_MAP_FILE = "../data/my_train_balanced.csv"
TEST_MAP_FILE = "../data/my_test_balanced.csv"


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data
    """
    revs = []

    sarc_train_file = data_folder[0]
    sarc_test_file = data_folder[1]

    # 读入训练集与测试集，现在只是映射集合，格式（对该评论的各个回复，评论id,各回复的讽刺与否标签）
    train_data = np.asarray(pd.read_csv(sarc_train_file, header=None))
    test_data = np.asarray(pd.read_csv(sarc_test_file, header=None))

    # 读取具体内容
    comments = json.loads(open(COMMENTS_FILE).read())
    vocab = defaultdict(float)

    for line in train_data:
        rev = []
        # 获得回复的标签label
        label_str = line[2]
        if label_str == 0:
            label = 0
        else:
            label = 1

        # 获得待检测目标的文本内容
        rev.append(comments[line[0]]['text'].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        # 词典编号
        for word in words:
            vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum = {"y": int(1),
                 "id": line[0],
                 "text": orig_rev,
                 "author": comments[line[0]]['author'],
                 "topic": comments[line[0]]['subreddit'],
                 "label": label,
                 "num_words": len(orig_rev.split()),
                 "split": int(1)}
        revs.append(datum)

    for line in test_data:
        rev = []
        label_str = line[2]
        if (label_str == 0):
            label = 0
        else:
            label = 1
        rev.append(comments[line[0]]['text'].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum = {"y": int(1),
                 "id": line[0],
                 "text": orig_rev,
                 "author": comments[line[0]]['author'],
                 "topic": comments[line[0]]['subreddit'],
                 "label": label,
                 "num_words": len(orig_rev.split()),
                 "split": int(0)}
        revs.append(datum)

    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def loadGloveModel(gloveFile, vocab):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        if word in vocab:
            model[word] = embedding

    print("Done.", len(model), " words loaded!")
    return model


def load_FastText(fname, vocab):
    """
    Loads 300x1 word vecs from Fasttext
    """
    print("Loading FastText Model")
    f = open(fname, 'r', encoding='UTF-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        if word in vocab:
            model[word] = embedding

    print("Done.", len(model), " words loaded!")
    return model


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  # 去除A-Za-z0-9(),!?'`外的字符
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
    string = re.sub(r"\s{2,}", " ", string)  # 两个以上连续的空白符，删除
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    # 词嵌入地址
    w2v_file = r"D:\code\pycharm\cascade_sarc\data\crawl-300d-2M.vec"
    # 训练集和测试集的映射地址
    data_folder = [TRAIN_MAP_FILE, TEST_MAP_FILE]
    print("loading data...")
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading FastText vectors...")
    w2v = load_FastText(w2v_file, vocab)
    print("FastText loaded!")
    print("num words already in FastText: " + str(len(w2v)))
    # 对于没有的词汇进行初始化
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open("mainbalancedpickle.p", "wb"))
    print("dataset created!")
    """
        loading data...
        data loaded!
        number of sentences: 219368
        vocab size: 74408
        max sentence length: 100
        loading FastText vectors...
        Loading FastText Model
        Done. 56856  words loaded!
        FastText loaded!
        num words already in FastText: 56856
        dataset created!
    """