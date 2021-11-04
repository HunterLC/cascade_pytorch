# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/18 21:10
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : TextCNN.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class TextCNN(nn.Module):
    def __init__(self, sequence_length=None, num_classes=2, vocab_size=None, word2vec_W=None, word_idx_map=None,
                 user_embeddings=None, topic_embeddings=None,
                 embedding_dim=300, batch_size=None, kernel_sizes=[3, 4, 5], num_kernel=128):
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size  # 已知词的数量
        self.embedding_dim = embedding_dim  # 每个词向量长度
        self.num_classes = num_classes  # 类别数
        self.num_channel = 1  # 输入的channel数
        self.num_kernel = num_kernel  # 每种卷积核的数量
        self.kernel_sizes = kernel_sizes  # 卷积核list，形如[2,3,4]

        # 嵌入层
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim)  # 词向量
        # self.embedding_chars = torch.index_select(self.W, 0, self.input_x)
        # self.embedding_chars_expanded = torch.unsqueeze(self.embedded_chars, -1)
        # self.embedding.weight = torch.nn.Parameter(word2vec_W)
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec_W))
        # 固定embedding，不参与更新
        self.embedding.weight.requires_grad = False

        # 用户嵌入
        self.user_embedding = nn.Embedding(user_embeddings.shape[0], user_embeddings.shape[1])  # 获得行列数
        self.user_embedding.weight.data.copy_(torch.from_numpy(user_embeddings))
        # 固定embedding，不参与更新
        self.user_embedding.weight.requires_grad = False

        # 主题嵌入
        self.topic_embedding = nn.Embedding(topic_embeddings.shape[0], topic_embeddings.shape[1])  # 获得行列数
        self.topic_embedding.weight.data.copy_(torch.from_numpy(topic_embeddings))
        # 固定embedding，不参与更新
        self.topic_embedding.weight.requires_grad = False

        # 卷积层
        self.convs = nn.ModuleList([nn.Conv2d(self.num_channel, self.num_kernel, (K, self.embedding_dim)) for K in kernel_sizes])
        # dropout层
        self.dropout = nn.Dropout(0.5)
        # 全连接层
        self.fc_layers1 = nn.Linear(len(self.kernel_sizes) * self.num_kernel, 100)
        self.fc_layers2 = nn.Linear(200 + 100, self.num_classes)

    def forward(self, x, user, topic):
        x = self.embedding(x)  # (N,W,D)
        x = x.unsqueeze(1)  # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat(x, 1)  # (N,Knum*len(Ks))

        x = self.fc_layers1(x)

        # 拼接用户嵌入和主题嵌入
        x = torch.cat((self.user_embedding(user), x, self.topic_embedding(topic)), 1)

        x = self.dropout(x)
        output = self.fc_layers2(x)
        return output


class SARCDataset(Dataset):
    def __init__(self, X, user, topic, y=None):
        self.data = torch.LongTensor(torch.from_numpy(X).long())
        self.user = torch.LongTensor(torch.from_numpy(user).long())
        self.topic = torch.LongTensor(torch.from_numpy(topic).long())
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.user[idx], self.topic[idx], self.label[idx]
        else:
            return self.data[idx], self.user[idx], self.topic[idx]

    def __len__(self):
        return len(self.data)
