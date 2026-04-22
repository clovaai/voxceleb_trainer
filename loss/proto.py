## Re-implementation of prototypical networks (https://arxiv.org/abs/1703.05175).
## Numerically checked against https://github.com/cyvius96/prototypical-network-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.test_normalize = False

        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised Prototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2
        
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
        label   = torch.arange(stepsize, device=x.device)
        nloss   = self.criterion(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1