# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/18 21:51
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : train_model.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import numpy as np
import pickle
import csv
from TextCNN import TextCNN, SARCDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd


def save(model, save_dir, save_prefix, steps):
    """
    保存模型
    :param model:
    :param save_dir:
    :param save_prefix:
    :param steps:
    :return:
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_()

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.data[0]
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f} acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy


print("loading data...QAQ")
x = pickle.load(open("./mainbalancedpickle.p", "rb"))
revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
print("data loaded!")  # Load data

# 载入用户嵌入
# print('loading wgcca embeddings...')
# wgcca_embeddings = np.load('../user/user_gcca_embeddings.npz')
# print('wgcca embeddings loaded')
#
# # 保存用户名，用来做索引
# ids = np.concatenate((np.array(["unknown"]), wgcca_embeddings['ids']), axis=0)
# # print(ids)
# user_embeddings = wgcca_embeddings['G']
# unknown_vector = np.random.normal(size=(1,100))
# user_embeddings = np.concatenate((unknown_vector, user_embeddings), axis=0)
# # 用户嵌入 (159884, 100)
# user_embeddings = user_embeddings.astype(dtype='float32')
#
# print(user_embeddings.shape)

print('loading wgcca embeddings...')
per_data = np.asarray(pd.read_csv('../user/personality/output/user_personality.csv', header=None))
print('wgcca embeddings loaded')
# numpy类型的字符串存在换行和若干空格，所以需要split
user_embeddings = [per_data[i][1].replace('[', '').replace(']', '').lstrip().rstrip().split() for i in range(len(per_data))]

user_embeddings = np.asarray(user_embeddings, dtype='float32')
# 保存用户名，用来做索引
ids = per_data[:, 0]
# print(ids)
# print(user_embeddings.shape)
# exit()
# 保存每位用户对应的index数值，例如 unknown是0，"Autistic_Buiscit"是1.....
wgcca_dict = {}
for i in range(len(ids)):
    wgcca_dict[ids[i]] = int(i)

# 载入主题嵌入
csv_reader = csv.reader(open("./../discourse/output/discourse_embeddings.csv"))
topic_embeddings = []
# 保存主题名
topic_ids = []
for line in csv_reader:
    topic_ids.append(line[0])
    topic_embeddings.append(line[1:])

topic_embeddings = np.asarray(topic_embeddings)
topic_embeddings_size = len(topic_embeddings[0])
topic_embeddings = topic_embeddings.astype(dtype='float32')
print(topic_embeddings.shape)
print("topic emb size: ", topic_embeddings_size)
# 保存每个主题对应的index数值
topics_dict = {}
for i in range(len(topic_ids)):
    try:
        topics_dict[topic_ids[i]] = int(i)
    except TypeError:
        print(i)

max_l = 100

x_text = []
author_text_id = []
topic_text_id = []
y = []

test_x = []
test_topic = []
test_author = []
test_y = []

for i in range(len(revs)):
    if revs[i]['split'] == 1:
        x_text.append(revs[i]['text'])
        try:
            # author_text_id.append(wgcca_dict['"'+revs[i]['author']+'"'])
            author_text_id.append(wgcca_dict[revs[i]['author']])
        except KeyError:
            author_text_id.append(0)
        try:
            topic_text_id.append(topics_dict[revs[i]['topic']])
        except KeyError:
            topic_text_id.append(0)
        temp_y = revs[i]['label']
        y.append(temp_y)
    else:
        test_x.append(revs[i]['text'])
        try:
            # test_author.append(wgcca_dict['"'+revs[i]['author']+'"'])
            test_author.append(wgcca_dict[revs[i]['author']])
        except:
            test_author.append(0)
        try:
            test_topic.append(topics_dict[revs[i]['topic']])
        except KeyError:
            test_topic.append(0)
        test_y.append(revs[i]['label'])
# 训练集label
y = np.asarray(y)
# 测试集label
test_y = np.asarray(test_y)
# print(len(author_text_id))
# exit()

# get word indices
x = []
for i in range(len(x_text)):
    x.append(np.asarray([word_idx_map[word] for word in x_text[i].split()]))

x_test = []
for i in range(len(test_x)):
    x_test.append(np.asarray([word_idx_map[word] for word in test_x[i].split()]))

# padding
for i in range(len(x)):
    if (len(x[i]) < max_l):
        x[i] = np.append(x[i], np.zeros(max_l - len(x[i])))
    elif (len(x[i]) > max_l):
        x[i] = x[i][0:max_l]
x = np.asarray(x)

for i in range(len(x_test)):
    if (len(x_test[i]) < max_l):
        x_test[i] = np.append(x_test[i], np.zeros(max_l - len(x_test[i])))
    elif (len(x_test[i]) > max_l):
        x_test[i] = x_test[i][0:max_l]
x_test = np.asarray(x_test)
y_test = test_y

topic_train = np.asarray(topic_text_id)
topic_test = np.asarray(test_topic)
author_train = np.asarray(author_text_id)
author_test = np.asarray(test_author)

# 词编号
word_idx_map["@"] = 0
# 编号词
rev_dict = {v: k for k, v in word_idx_map.items()}

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
batch_size = 4096

train_set = SARCDataset(x, author_train, topic_train, y)
test_set = SARCDataset(x_test, author_test, topic_test, y_test)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                          pin_memory=True)  # only shuffle the training data
valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


def train():
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = TextCNN(sequence_length=max_l, num_classes=2, vocab_size=len(vocab), word2vec_W=W,
                    word_idx_map=word_idx_map, user_embeddings=user_embeddings, topic_embeddings=topic_embeddings,
                    embedding_dim=300, batch_size=None, kernel_sizes=[3, 4, 5], num_kernel=128).to(device)
    model.device = device

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)

    # The number of training epochs.
    n_epochs = 100

    best_acc = 0
    n_epochs_train_loss = []
    n_epochs_train_accs = []
    n_epochs_valid_loss = []
    n_epochs_valid_accs = []

    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of data and corresponding labels.
            data, user, topic, labels = batch
            # print(data)
            # print(labels)
            # exit()

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(data.to(device), user.to(device), topic.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            # print(logits.size())
            # print(labels.size())
            # break
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        n_epochs_train_loss.append(train_loss)
        n_epochs_train_accs.append(train_acc)


        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            data, user, topic, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(data.to(device), user.to(device), topic.to(device))

            # We can still compute the loss (but not the gradient).
            # print(logits.size())
            # print(labels.size())
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        n_epochs_valid_loss.append(valid_loss)
        n_epochs_valid_accs.append(valid_acc)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        # save model
        if epoch % n_epochs == 0:  # first epoch
            best_acc = valid_acc
            save(model, './output', 'best', epoch)
        else:
            if valid_acc > best_acc:
                best_acc = valid_acc
                save(model, './output', 'best', epoch)
    print('best acc is {}'.format(best_acc))
    draw(save_dir='./output', train_loss=n_epochs_train_loss, train_accs=n_epochs_train_accs,
         valid_loss=n_epochs_valid_loss, valid_accs=n_epochs_valid_accs)


def draw(save_dir='./output', train_loss=None, train_accs=None, valid_loss=None, valid_accs=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # draw loss
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fname=save_dir + '/loss.jpg', figsize=[10, 10])

    # draw acc
    plt.figure()
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, 'bo', label='Training acc')
    plt.plot(epochs, valid_accs, 'b', label='validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(fname=save_dir + '/acc.jpg', figsize=[10, 10])
    plt.show()


train()
