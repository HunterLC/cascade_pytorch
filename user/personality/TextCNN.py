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
    def __init__(self, sequence_length=None, num_classes=2, vocab_size=None, word2vec_W=None,
                 embedding_dim=300, kernel_sizes=[3, 4, 5], num_kernel=128):
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size  # 已知词的数量
        self.embedding_dim = embedding_dim  # 每个词向量长度
        self.num_classes = num_classes  # 类别数
        Ci = 1  # 输入的channel数
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

        # 卷积层
        self.convs = nn.ModuleList([nn.Conv2d(Ci, self.num_kernel, (K, self.embedding_dim)) for K in kernel_sizes])
        # dropout层
        self.dropout = nn.Dropout(0.5)
        # 全连接层
        self.fc_layers1 = nn.Linear(len(self.kernel_sizes) * self.num_kernel, 100)
        self.fc_layers2 = nn.Linear(100, self.num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (N,W,D)
        x = x.unsqueeze(1)  # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat(x, 1)  # (N,Knum*len(Ks))

        fc_layers1 = self.fc_layers1(x)

        output = self.dropout(fc_layers1)
        output = self.fc_layers2(output)

        return fc_layers1, output



class PersonalityDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.LongTensor(torch.from_numpy(X).long())
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)
