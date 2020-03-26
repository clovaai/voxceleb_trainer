#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class SoftmaxLoss(nn.Module):
	def __init__(self, in_feats, n_classes=10):
	    super(SoftmaxLoss, self).__init__()
	    
	    self.criterion  = torch.nn.CrossEntropyLoss()
	    self.fc 		= nn.Linear(in_feats,n_classes)

	    print('Initialised Softmax Loss')

	def forward(self, x, label=None):

		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1, _ = accuracy(x.detach().cpu(), label.detach().cpu(), topk=(1, 5))

		return nloss, prec1