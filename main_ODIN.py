# Adapted from https://github.com/facebookresearch/odin

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
parser.add_argument('--model_num', default="all", type=str,
                    help='which model to test')
parser.add_argument('--magnitude', default=0.0014, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')
parser.add_argument('--device', default = 'cpu', type = str,
		    help='cpu device')
parser.add_argument('--batch_size', default = 64, type = int,
		    help='batch size bud')
parser.add_argument('--get_metrics_only', action=argparse.BooleanOptionalAction,
		    help='only include if all the desired softmax scores have already been calculated')
parser.set_defaults(argument=True)

# Setting the name of neural networks (nnName)
# Custom CNN trained on MNIST: BASELINE_CNN
# Custom CNN trained on CIFAR10: ADVANCED_CNN

# Setting the perturbation magnitude
# magnitude = 0.0014

# Setting the temperature
# temperature = 1000

# If you have already collected all the confidence scores, 
# set get_metrics_only = True to go straight to evaluation
# get_metrics_only = True

def main():
    global args
    args = parser.parse_args()
    c.test(args.nn, args.device, args.get_metrics_only, args.model_num, args.magnitude, args.temperature, args.batch_size)

if __name__ == '__main__':
    main()

















