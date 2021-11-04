# -*- coding: utf-8 -*-

"""
@Software: Win11  Python 3.6 |Anaconda
@IDE--Env: PyCharm--
@Time    : 2021/10/23 14:51
@Author  : HunterLC
@connect : lc411887055@gmail.com
@Project : cascade_sarc
@File    : train_personality.py
@Version : 1.0.0 
@Desc    :
@LastTime: 
"""

import pickle
import numpy as np

np.random.seed(10)

from user.personality.TextCNN import TextCNN, PersonalityDataset
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


# Data Preparation
# ==================================================

print("loading data...", end=' ')
x = pickle.load(open("personalitypickle.p", "rb"))
revs, W, word_idx_map, vocab_len = x[0], x[1], x[2], x[3]
print("data loaded!")  # Load data

x_text = np.asarray([revs[i]['text'] for i in range(len(revs))])
y = np.asarray([revs[i]['Label'] for i in range(len(revs))])

x = []
for i in range(len(x_text)):
    x.append([word_idx_map[word] for word in x_text[i].split()])

for i in range(len(x)):
    if len(x[i]) < 1000:
        x[i] = np.append(x[i], np.zeros(1000 - len(x[i])))
    elif len(x[i]) > 1000:
        x[i] = x[i][0:1000]
x = np.asarray(x)

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(0.1 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
x_train = np.asarray(x_train)
x_dev = np.asarray(x_dev)
y_train = np.asarray(y_train)
y_dev = np.asarray(y_dev)
print(x_dev)
print(y_dev)

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
batch_size = 64

train_set = PersonalityDataset(x_train, y_train)
test_set = PersonalityDataset(x_dev, y_dev)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0,
                          pin_memory=True)  # only shuffle the training data
valid_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0, pin_memory=True)


def train():
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = TextCNN(sequence_length=1000, num_classes=5, vocab_size=vocab_len, word2vec_W=W,
                    embedding_dim=300, kernel_sizes=[3, 4, 5], num_kernel=128).to(device)
    model.device = device

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

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
            data, labels = batch

            # Forward the data. (Make sure data and model are on the same device.)
            _, logits = model(data.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            # print(logits.size())
            # print(labels.size())
            # break
            # labels = labels.squeeze()
            # print(logits.size())
            # print(labels.size())
            # break
            loss = criterion(logits, torch.argmax(labels.to(device), dim=1))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == torch.argmax(labels.to(device), dim=1)).float().mean()

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
            data, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                _, logits = model(data.to(device))

            # We can still compute the loss (but not the gradient).
            # print(logits.size())
            # print(labels.size())
            # labels = labels.squeeze()
            loss = criterion(logits, torch.argmax(labels.to(device), dim=1))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == torch.argmax(labels.to(device), dim=1)).float().mean()

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
