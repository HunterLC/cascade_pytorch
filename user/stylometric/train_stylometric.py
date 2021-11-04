# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/18 20:28
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : train_stylometric.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import os
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

# gensim里Doc2vec模型需要的输入为固定格式，输入样本为：[句子，句子序号],这里需要用gensim中Doc2vec里的TaggedDocument来包装输入的句子
TaggedDocument = gensim.models.doc2vec.TaggedDocument

# Input file path
USER_PARAGRAPH_INPUTS = "./output/train_balanced_user.csv"

"""
    构造gensim里Doc2vec模型需要数据格式
"""


class LabeledLineSentence(object):
    def __init__(self, doc_list, label_list):
        self.doc_list = doc_list
        self.label_list = label_list

    def __iter__(self):
        count = 0
        for idx, doc in enumerate(self.doc_list):
            # yield作用类似于return，但是会记住当前迭代的位置，下次迭代会从上次记住的位置继续开始
            try:
                yield TaggedDocument(doc.split(), tags=[self.label_list[idx]])
            except AttributeError:
                count += 1


def train_doc2vec(dataset=None):
    """
    训练doc2vec并保存训练的模型

    Doc2Vec参数
    size 是特征向量的纬度。

    window 是要预测的词和文档中用来预测的上下文词之间的最大距离。

    alpha 是初始化的学习速率，会随着训练过程线性下降。

    seed 是随机数生成器。.需要注意的是，对于一个完全明确的重复运行（fully deterministically-reproducible run），你必须同时限制模型单线程工作以消除操作系统线程调度中的有序抖动。（在python3中，解释器启动的再现要求使用PYTHONHASHSEED环境变量来控制散列随机化）

    min_count 忽略总频数小于此的所有的词。

    max_vocab_size 在词汇累积的时候限制内存。如果有很多独特的词多于此，则将频率低的删去。每一千万词类大概需要1G的内存，设为None以不限制（默认）。

    sample 高频词被随机地降低采样的阈值。默认为0（不降低采样），较为常用的事1e-5。

    dm 定义了训练的算法。默认是dm=1,使用 ‘distributed memory’ (PV-DM)，否则 distributed bag of words (PV-DBOW)。

    workers 使用多少现成来训练模型（越快的训练需要越多核的机器）。

    iter 语料库的迭代次数。从Word2Vec中继承得到的默认是5，但在已经发布的‘Paragraph Vector’中，设为10或者20是很正常的。

    hs 如果为1 (默认)，分层采样将被用于模型训练（否则设为0）。

    negative 如果 > 0，将使用负采样，它的值决定干扰词的个数（通常为5-20）。

    dm_mean 如果为0（默认），使用上下文词向量的和；如果为1，使用均值。（仅在dm被用在非拼接模型时使用）

    dm_concat 如果为1，使用上下文词向量的拼接，默认是0。注意，拼接的结果是一个更大的模型，输入的大小不再是一个词向量（采样或算术结合），而是标签和上下文中所有词结合在一起的大小。

    dm_tag_count 每个文件期望的文本标签数，在使用dm_concat模式时默认为1。

    dbow_words 如果设为1，训练word-vectors (in skip-gram fashion) 的同时训练 DBOW doc-vector。默认是0 (仅训练doc-vectors时更快)。

    trim_rule 词汇表修建规则，用来指定某个词是否要被留下来。被删去或者作默认处理 (如果词的频数< min_count则删去)。可以设为None (将使用min_count)，或者是随时可调参 (word, count, min_count) 并返回util.RULE_DISCARD,util.RULE_KEEP ,util.RULE_DEFAULT之一。注意：这个规则只是在build_vocab()中用来修剪词汇表，而且没被保存。

    """
    assert dataset

    data = np.asarray(pd.read_csv(dataset, header=None))
    # 获取label列表
    list_doc_labels = [data[i][0] for i in range(data.shape[0])]
    # 获取主题下的内容
    list_doc_contents = [data[i][1] for i in range(data.shape[0])]
    # 构造训练集
    it = LabeledLineSentence(list_doc_contents, list_doc_labels)

    model = gensim.models.Doc2Vec(vector_size=100, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)

    # 定义输出目录
    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 迭代训练
    for epoch in tqdm(range(1)):
        # print(epoch)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
        model.train(it, total_examples=model.corpus_count, epochs=model.iter)
        # 保存模型
        model.save(directory + "/user_stylometric")


train_doc2vec(USER_PARAGRAPH_INPUTS)
