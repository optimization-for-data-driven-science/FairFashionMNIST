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

for i in range(50):
	C1 = V(torch.zeros(5, 1, 3, 3), requires_grad=False)
	C2 = V(torch.zeros(10, 5, 3, 3), requires_grad=False)
	F1 = V(torch.zeros(5 * 5 * 10, 100), requires_grad=False)
	F2 = V(torch.zeros(100, 10), requires_grad=False)
	torch.nn.init.xavier_normal_(C1.data)
	torch.nn.init.xavier_normal_(C2.data)
	torch.nn.init.xavier_normal_(F1.data)
	torch.nn.init.xavier_normal_(F2.data)
	BC1 = V(torch.randn(5) * 1 / 8, requires_grad=False)
	BC2 = V(torch.randn(10) * 1 / 16, requires_grad=False)
	BF1 = V(torch.randn(100) * 1 / 100, requires_grad=False)
	BF2 = V(torch.randn(10) * 1 / 10, requires_grad=False)
	w = Weights(C1.data, C2.data, F1.data, F2.data, BC1.data, BC2.data, BF1.data, BF2.data)
	fname = "init_%d.ckpt" % i
	with open(fname, "wb") as f:
		pickle.dump(w, f)
		print("\nWeights saved to %s!\n" % fname)


