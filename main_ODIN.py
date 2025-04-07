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
import argparse
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import time
#import lmdb
from scipy import misc
import ODIN.cal as c


parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--nn', default="BASELINE_CNN", type=str,
                    help='neural network name and training set')
parser.add_argument('--magnitude', default=0.0014, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')
parser.add_argument('--device', default = 'cpu', type = str,
		    help='cpu device')
parser.set_defaults(argument=True)


### TO DO:
#  add support for advanced model 
# add support for near and far OOD datasets
# check the metrics code
# write some code to try different temperature and perturbation values?

# Setting the name of neural networks (nnName)
# Custom CNN trained on MNIST: BASELINE_CNN
# Custom CNN trained on CIFAR10: ADVANCED_CNN

# Setting the name of the out-of-distribution dataset (out_dataset)
# FashionMNIST: FashionMNIST
# EMNIST: EMNIST

# Setting the perturbation magnitude
# magnitude = 0.0014

# Setting the temperature
# temperature = 1000

def main():
    global args
    args = parser.parse_args()
    c.test(args.nn, args.device, args.magnitude, args.temperature)

if __name__ == '__main__':
    main()

















