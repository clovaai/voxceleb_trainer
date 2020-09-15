#! /usr/bin/python
# -*- encoding: utf-8 -*-

from models.ResNetSE34L import *
from models.ResNetBlocks import *



def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model
