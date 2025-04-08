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
from Data import FashionMNIST, EMNIST, SVHN, CIFAR10, MNIST, CIFAR100
from Model_Architecture.Adv_CNN import ResNet18
from . import calMetric as m
from . import calData as d
from .baseline_cnn_logits import BaselineCNN

start = time.time()

transform = transforms.Compose([transforms.ToTensor()])
criterion = nn.CrossEntropyLoss()

def test(nnName, CUDA_DEVICE, metrics_only, model_num, epsilon, temperature):
    if nnName == "BASELINE_CNN" : 
        testsetFarOutData = FashionMNIST(batch_size=1)
        testloaderFarOut = testsetFarOutData.get_test()
        
        testsetNearOutData = EMNIST(batch_size=1)
        testloaderNearOut = testsetNearOutData.get_test()
        
        testsetInData = MNIST(batch_size=1)
        testloaderIn = testsetInData.get_test()
        
    if nnName == "ADVANCED_CNN" : 
        testsetFarOutData = SVHN(batch_size=1)
        testloaderFarOut = testsetFarOutData.get_test()
        
        testsetNearOutData = CIFAR100(batch_size=1)
        testloaderNearOut = testsetNearOutData.get_test()
        
        testsetInData = CIFAR10(batch_size=1)
        testloaderIn = testsetInData.get_test()
    
    if model_num == "all":
        model_list = ["1","2","3","4","5"]
    else:
        model_list = [model_num]
    
    for n in model_list:
        if nnName == "BASELINE_CNN" : 
            net1 = BaselineCNN()
            model_name = 'baseline_cnn_'
        elif nnName == "ADVANCED_CNN" :
            net1 = ResNet18()
            model_name = 'adv_cnn_'
        print("Processing model number: ", n)
        model_name = model_name + n

        net1.load_state_dict(torch.load(f"./Saved_Models/{model_name}.pth", map_location=CUDA_DEVICE))
        net1.to(CUDA_DEVICE)

        if not metrics_only:
            print("Calculating softmax scores")
            d.testData(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderNearOut, testloaderFarOut, nnName, n, epsilon, temperature) 
        m.new_metric(nnName, n)

        # delete net1
        del net1








