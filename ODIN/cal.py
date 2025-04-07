# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
from baseline_cnn import BaselineCNN

start = time.time()

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
# ])
transform = transforms.Compose([transforms.ToTensor()])
criterion = nn.CrossEntropyLoss()

def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    if nnName == "custom" : 
        net1 = BaselineCNN()
        model_name = 'baseline_cnn_1'
        net1.load_state_dict(torch.load(f"../Saved_Models/{model_name}.pth"))
    else:
        net1 = torch.load("../models/{}.pth".format(nnName))

    net1.to(CUDA_DEVICE)

    if nnName == "custom" : 
        testsetout = torchvision.datasets.FashionMNIST(root='../Data/', train=False, download=True, transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                            shuffle=False)
        testset = torchvision.datasets.MNIST(root='../Data/', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False)
    
    d.testData(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature) 
    m.metric(nnName, dataName)








