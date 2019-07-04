import time
import json
import torch
import pickle
import xlsxwriter
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.autograd import Variable as V
from mnist_reader import load_mnist

X_train, y_train = load_mnist(kind='train')
X_test, y_test = load_mnist(kind='t10k')

X_train =  X_train[np.any([y_train == 4, y_train == 6, y_train == 0], axis=0)]
y_train =  y_train[np.any([y_train == 4, y_train == 6, y_train == 0], axis=0)]
X_test =  X_test[np.any([y_test == 4, y_test == 6, y_test == 0], axis=0)] 
y_test =  y_test[np.any([y_test == 4, y_test == 6, y_test == 0], axis=0)] 

X_train_tmp = {}
y_train_tmp = {}
X_train_tmp[4] = []
X_train_tmp[6] = []
X_train_tmp[0] = []
y_train_tmp[4] = []
y_train_tmp[6] = []
y_train_tmp[0] = []

for i in range(18000):
	X_train_tmp[y_train[i]].append(X_train[i].tolist())
	y_train_tmp[y_train[i]].append(y_train[i])

X_train2 = []
y_train2 = []

for i in range(30):
	k = 200 * i
	X_train2 = X_train2 + X_train_tmp[4][k: k + 200] + X_train_tmp[6][k: k + 200] + X_train_tmp[0][k: k + 200]
	y_train2 = y_train2 + y_train_tmp[4][k: k + 200] + y_train_tmp[6][k: k + 200] + y_train_tmp[0][k: k + 200]

# X_train2 = X_train_tmp[4] + X_train_tmp[6] + X_train_tmp[0]
# y_train2 = y_train_tmp[4] + y_train_tmp[6] + y_train_tmp[0]

X_train_ = np.array(X_train2)
y_train_ = np.array(y_train2)

Data = []
Data.append(X_train_)
Data.append(y_train_)
Data.append(X_test)
Data.append(y_test)

with open('fashion_mnist_three.pkl', 'wb') as f:
	pickle.dump(Data, f)
