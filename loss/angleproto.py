#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class AngleProtoLoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(AngleProtoLoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1).expand(-1,-1,stepsize),out_anchor.unsqueeze(-1).expand(-1,-1,stepsize).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label       = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss       = self.criterion(cos_sim_matrix, label)
        prec1, _    = accuracy(cos_sim_matrix.detach().cpu(), label.detach().cpu(), topk=(1, 5))

        return nloss, prec1