#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import pickle

train_size = 8000
test_size = 2000

input_size = 20
hidden_sizes = [50, 50]
output_size = 1
num_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)

if not os.path.exists('../data/X_train.pk'):
    X_train = np.random.randn(train_size, input_size).astype(np.float32)
    with open('../data/X_train.pk', 'wb') as p:
        pickle.dump(X_train, p)
else:
    with open('../data/X_train.pk', 'rb') as p:
        X_train = pickle.load(p)

if not os.path.exists('../data/X_test.pk'):
    X_test = np.random.randn(test_size, input_size).astype(np.float32)
    with open('../data/X_test.pk', 'wb') as p:
        pickle.dump(X_test, p)
else:
    with open('../data/X_test.pk', 'rb') as p:
        X_test = pickle.load(p)

if not os.path.exists('../data/y_train.pk'):
    y_train = np.random.randint(num_classes, size=train_size)
    with open('../data/y_train.pk', 'wb') as p:
        pickle.dump(y_train, p)
else:
    with open('../data/y_train.pk', 'rb') as p:
        y_train = pickle.load(p)

if not os.path.exists('../data/y_test.pk'):
    y_test = np.random.randint(num_classes, size=train_size)
    with open('../data/y_test.pk', 'wb') as p:
        pickle.dump(y_test, p)
else:
    with open('../data/y_test.pk', 'rb') as p:
        y_test = pickle.load(p)


print('Shape of X_train:', X_train.shape)
print('Shape of X_train:', X_test.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of y_test:', y_test.shape)

#class SimpleModel(nn.Module):
#    def __init__(self, input_size, hidden_sizes, output_size):
#        super(SimpleModel, self).__init__()
#        self.input_size = input_size
#        self.output_size = output_size
#        self.fcs = []  # List of fully connected layers
#        in_size = input_size
#
#        for i, next_size in enumerate(hidden_sizes):
#            fc = nn.Linear(in_features=in_size, out_features=next_size)
#            in_size = next_size
#            self.__setattr__('fc{}'.format(i), fc)  # set name for each fullly connected layer
#            self.fcs.append(fc)
#
#        self.last_fc = nn.Linear(in_features=in_size, out_features=output_size)
#
#    def forward(self, x):
#        for i, fc in enumerate(self.fcs):
#            x = fc(x)
#            x = nn.ReLU()(x)
#        out = self.last_fc(x)
#        return nn.Sigmoid()(out)
#
#
#model_pytorch = SimpleModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
#model_pytorch.load_state_dict(torch.load('./models/model_simple.pt'))
#print(model_pytorch)

model_onnx = onnx.load('../models/model_simple.onnx')
tf_rep = prepare(model_onnx)
