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
import numpy as np

def get_softmax_outputs(logits):
    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = logits.data.cpu().numpy()[0]
    # Calculate softmax scores, avoid numerical overflow
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
    return nnOutputs

def get_gradient(nnOutputs, CUDA_DEVICE, criterion, inputs, outputs):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    maxIndexTemp = np.argmax(nnOutputs)
    labels = Variable(torch.LongTensor([maxIndexTemp]).to(CUDA_DEVICE))
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    return gradient

def get_ODIN_output(net1, outputs, softmaxed_outputs, inputs, criterion, CUDA_DEVICE, noiseMagnitude1, temper):
    outputs = outputs / temper
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    gradient = get_gradient(softmaxed_outputs, CUDA_DEVICE, criterion, inputs, outputs)
    # Adding small perturbations to images
    perturbed_inputs = torch.add(inputs.data, gradient, alpha =-noiseMagnitude1)
    perturbed_outputs = net1(Variable(perturbed_inputs))
    perturbed_outputs = perturbed_outputs / temper
    return perturbed_outputs

def get_score(net1, criterion, CUDA_DEVICE, dataloader, noiseMagnitude1, temper, conf_baseline, conf_ODIN):
    N = 10000
    t0 = time.time()
    for j, data in enumerate(dataloader):
        images, _ = data
        
        inputs = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs) # logits

        #### BASELINE ####
        # Confidence before perturbation and temperature scaling:
        softmaxed_outputs = get_softmax_outputs(outputs) # apply softmax to logit outputs
        conf_baseline.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(softmaxed_outputs))) 

        #### ODIN ####
        # Using temperature scaling on the logit outputs
        ODIN_outputs = get_ODIN_output(net1, outputs, softmaxed_outputs, inputs, criterion, CUDA_DEVICE, noiseMagnitude1, temper)
        # Calculating the confidence after adding perturbations and temperature scaling
        softmaxed_outputs = get_softmax_outputs(ODIN_outputs)
        conf_ODIN.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(softmaxed_outputs)))

        if j % 1000 == 999:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1, N, time.time()-t0))
            t0 = time.time()
        
        if j == N - 1: break

def testData(net1, criterion, CUDA_DEVICE, testloader_ID, testloader_OOD, nnName, dataName, noiseMagnitude1, temper):
    confidences_baseline_ID = open("./softmax_scores/confidence_Base_In.txt", 'w')
    confidences_baseline_OOD = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    confidences_ODIN_ID = open("./softmax_scores/confidence_Our_In.txt", 'w')
    confidences_ODIN_OOD = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    
    print("Processing in-distribution images")
    get_score(net1, criterion, CUDA_DEVICE, testloader_ID, noiseMagnitude1, temper, confidences_baseline_ID, confidences_ODIN_ID)

    print("Processing out-of-distribution images")
    get_score(net1, criterion, CUDA_DEVICE, testloader_OOD, noiseMagnitude1, temper, confidences_baseline_OOD, confidences_ODIN_OOD)






