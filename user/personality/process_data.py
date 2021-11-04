# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/22 16:45
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : process_data.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import numpy as np
import pickle
from collections import defaultdict
import sys
import re
import pandas as pd

# noinspection PyCompatibility
from builtins import range


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data
    """
    revs = []
    per_file = data_folder[0]
    fb_file = data_folder[1]
    vocab = defaultdict(float)

    train_data = np.asarray(pd.read_csv(per_file, header=None, encoding="cp1252"))
    train_data = train_data[1:]
    users = train_data[:, 0]
    #    print train_data[1][1]
    paragraphs = train_data[:, 1]
    Labels = []
    for i in range(len(train_data)):
        label = []
        for j in range(2, 7, 1):
            if (train_data[i][j] == 'y'):
                label.append(1)
            else:
                label.append(0)
        Labels.append(label)

    df_fb_train_data = pd.read_csv(fb_file, header=0, usecols=[0, 1, 7, 8, 9, 10, 11], encoding="cp1252")
    fb_train_data = np.asarray(df_fb_train_data.groupby(df_fb_train_data['X.AUTHID']).agg(lambda x: ' '.join(x)))
    # 创建数据集

    fb_paragraphs = fb_train_data[:, 1]
    fb_Labels = []
    for i in range(len(fb_train_data)):
        fb_Label = []
        for j in range(2, 7, 1):
            if 'y' in train_data[i][j]:
                fb_Label.append(1)
            else:
                fb_Label.append(0)
        fb_Labels.append(fb_Label)

    paragraphs = np.append(paragraphs, fb_paragraphs)
    print(paragraphs.shape)
    Labels = np.append(Labels, fb_Labels, 0)
    print(Labels.shape)

    ind = 0
    for line in paragraphs:
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {"y": 1,
                 "text": orig_rev,
                 "Label": Labels[ind],
                 "num_words": len(orig_rev.split()),
                 "split": np.random.randint(0, cv)}
        revs.append(datum)
        ind += 1
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


def load_fasttext(fname, vocab):
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
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    w2v_file = r"D:\code\pycharm\cascade_sarc\data\crawl-300d-2M.vec"
    data_folder = [r"D:\code\pycharm\cascade_sarc\data\personality_essay.csv",
                   r"D:\code\pycharm\cascade_sarc\data\my_personality_train.csv"]

    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...", end=' ')
    w2v = load_fasttext(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    # rand_vecs = {}
    # add_unknown_words(rand_vecs, vocab)
    # W2, _ = get_W(rand_vecs)
    pickle.dump([revs, W, word_idx_map, len(vocab)], open("personalitypickle.p", "wb"))
    print("dataset created!")
