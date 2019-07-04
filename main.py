import time
import json
import torch
import pickle
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.autograd import Variable as V
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Weights:
     def __init__(self, C1_, C2_, F1_, F2_, BC1_, BC2_, BF1_, BF2_):
          self.C1 = C1_
          self.C2 = C2_
          self.F1 = F1_
          self.F2 = F2_
          self.BC1 = BC1_
          self.BC2 = BC2_
          self.BF1 = BF1_
          self.BF2 = BF2_


accAll = []
lossAll = []

for numRun in range(50):

	start = time.time()	

	beta = 0.1

	with open("init_%d.ckpt" % numRun, "rb") as f:
		W = pickle.load(f)

	C1 = V(W.C1, requires_grad=True)
	C2 = V(W.C2, requires_grad=True)
	F1 = V(W.F1, requires_grad=True)
	F2 = V(W.F2, requires_grad=True)
	BC1 = V(W.BC1, requires_grad=True)
	BC2 = V(W.BC2, requires_grad=True)
	BF1 = V(W.BF1, requires_grad=True)
	BF2 = V(W.BF2, requires_grad=True)

	with open('fashion_mnist_three.pkl', 'rb') as f:
		Data = pickle.load(f)

	Xtr = torch.tensor(Data[0])
	Ytr = torch.tensor(Data[1])
	Xte = torch.tensor(Data[2])
	Yte = torch.tensor(Data[3])

	Xtrain = Xtr.float().view(18000, 1, 28, 28)
	Xtest = Xte.float().view(3000, 1, 28, 28)

	Ytrain = Ytr
	Ytest = Yte

	batch_size = 6000 * 3

	Iter = [5500]

	var_list = [C1, C2, F1, F2, BC1, BC2, BF1, BF2]
	loss_ = []

	for MaxIter, step_size in zip(Iter, ss):
		
		optimizer = torch.optim.SGD(var_list, lr=step_size)
		
		for i in range(MaxIter):



			X = Xtrain.cuda()
			y = Ytrain.cuda().long()

			tmp = F.conv2d(X, C1, bias=BC1)
			tmp = F.tanh(tmp)
			tmp = F.max_pool2d(tmp, 2, 2)
			tmp = F.conv2d(tmp, C2, bias=BC2)
			tmp = F.tanh(tmp)
			tmp = F.max_pool2d(tmp, 2, 2)
			tmp = F.tanh(tmp.view(batch_size, 5 * 5 * 10).mm(F1) + BF1)
			pred = tmp.mm(F2) + BF2

			loss = (-F.log_softmax(pred)).gather(1, y.view(-1, 1)).view(3, -1)
			loss, _ = torch.sort(loss.mean(dim=1))
			a = V(loss.data.cpu(), requires_grad=False)

			flag = False
			for l in reversed(range(n_cate)):
				lambda_ = a[l]
				t = F.relu((a - lambda_) / beta)
				if t.sum() > 1.0:
					lambda_ = (a[l + 1:].sum() - beta) / (n_cate - l - 1)
					t = F.relu((a - lambda_) / beta)
					flag = True
					break

			if (flag == False):
				lambda_ = (a.sum() - beta) / n_cate
				t = F.relu((a - lambda_) / beta)

			loss = (loss * t.cuda()).sum() - t.cuda().pow(2).sum() * beta / 2


			optimizer.zero_grad()
		          
			loss.backward()

			optimizer.step()

			loss_.append(loss.item())


			if ((i) % 100 == 0):
				tmp2 = F.conv2d(Xtest.cuda(), C1, bias=BC1)
				tmp2 = F.tanh(tmp2)
				tmp2 = F.max_pool2d(tmp2, 2, 2)
				tmp2 = F.conv2d(tmp2, C2, bias=BC2)
				tmp2 = F.tanh(tmp2)
				tmp2 = F.max_pool2d(tmp2, 2, 2)
				tmp2 = F.tanh(tmp2.view(3000, 5 * 5 * 10).mm(F1) + BF1)
				pred = tmp2.mm(F2) + BF2
				predicted = pred.max(1)[1].int()
				accuracy = 1 - (Ytest.int().cuda() - predicted).abs().clamp(min=0, max=1).sum().float() / 3000
				print("Iteration %d done, loss = %.8f, accuracy = %.4f" % (i, loss.item(), accuracy))

	acc = [0] * 10
	for i in range(3000):
		if (predicted[i].item() == Yte[i]):
			acc[Yte[i]] += 1
	print(acc)
	accAll.append(acc)
	end = time.time()
	print("Run %d done in %.0f min(s)" % (numRun, (end - start) / 60))
	lossAll.append(loss_)

accAll = torch.tensor(accAll).float()
mean = accAll.mean(dim=0)
std = accAll.std(dim=0)
print("mean", accAll.mean(dim=0))
print("std", accAll.std(dim=0))

with open("sgd_tanh_minmax_%.6f.csv" % beta, "w") as f:
	accAll_str = str(accAll.tolist()).replace("[", "")
	accAll_str = accAll_str.replace("]]", "")
	accAll_str = accAll_str.replace("],", "\n")
	# accAll_str = accAll_str.replace("],", "\n")
	f.write(accAll_str)
	f.write("\n\nmean\n")
	mean_str = str(mean.tolist()).replace("[", "")
	mean_str = mean_str.replace("]", "")
	mean_str = mean_str.replace("],", "\n")
	# mean_str = mean_str.replace("],", "\n")
	f.write(mean_str)
	f.write("\nstd\n")
	std_str = str(std.tolist()).replace("[", "")
	std_str = std_str.replace("]", "")
	std_str = std_str.replace("],", "\n")
	# std_str = std_str.replace("],", "\n")
	f.write(std_str)

with open("adam_tanh_minmax_%.6f.loss" % beta, "wb") as f:
	pickle.dump(lossAll, f)



