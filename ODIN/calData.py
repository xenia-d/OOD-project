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
    nnOutputs = logits.data.cpu().numpy()#[0]
    individual_nnOutputs = []
    for output in nnOutputs:
        # Calculate softmax scores, avoid numerical overflow
        output = output - np.max(output)
        output = np.exp(output)/np.sum(np.exp(output))
        individual_nnOutputs.append(output)
    return individual_nnOutputs

def get_gradient(nnOutputs, CUDA_DEVICE, criterion, inputs, outputs):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    maxIndexTemp = np.argmax(nnOutputs, axis=1)
    labels = Variable(torch.LongTensor(maxIndexTemp).to(CUDA_DEVICE))
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
        for output in softmaxed_outputs:
            # print("Full Output: ", output, " - max:", np.max(output))
            conf_baseline.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(output))) 

        #### ODIN ####
        # Using temperature scaling on the logit outputs
        ODIN_outputs = get_ODIN_output(net1, outputs, softmaxed_outputs, inputs, criterion, CUDA_DEVICE, noiseMagnitude1, temper)
        # Calculating the confidence after adding perturbations and temperature scaling
        softmaxed_outputs = get_softmax_outputs(ODIN_outputs)
        for output in softmaxed_outputs:
            conf_ODIN.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(output)))

        if j % 100 == 99:
            print("{:4}/{:4} batches processed, {:.1f} seconds used.".format((j+1), N, time.time()-t0))
            t0 = time.time()
        
        if j == N - 1: break

def testData(net1, criterion, CUDA_DEVICE, testloader_ID, testloader_near_OOD, testloader_far_OOD, nnName, n, noiseMagnitude1, temper):
    if nnName == "BASELINE_CNN":
        folder_name = 'baseline_cnn'
    elif nnName == "ADVANCED_CNN":
        folder_name = 'adv_cnn'
    
    confidences_baseline_ID = open(f"ODIN/softmax_scores/{folder_name}/{n}_confidence_Base_In.txt", 'w')
    confidences_baseline_Near_OOD = open(f"ODIN/softmax_scores/{folder_name}/{n}_confidence_Base_Near_Out.txt", 'w')
    confidences_baseline_Far_OOD = open(f"ODIN/softmax_scores/{folder_name}/{n}_confidence_Base_Far_Out.txt", 'w')
    
    confidences_ODIN_ID = open(f"ODIN/softmax_scores/{folder_name}/{n}_confidence_ODIN_In.txt", 'w')
    confidences_ODIN_Near_OOD = open(f"ODIN/softmax_scores/{folder_name}/{n}_confidence_ODIN_Near_Out.txt", 'w')
    confidences_ODIN_Far_OOD = open(f"ODIN/softmax_scores/{folder_name}/{n}_confidence_ODIN_Far_Out.txt", 'w')
    
    print("Processing in-distribution images")
    get_score(net1, criterion, CUDA_DEVICE, testloader_ID, noiseMagnitude1, temper, confidences_baseline_ID, confidences_ODIN_ID)

    print("Processing near out-of-distribution images")
    get_score(net1, criterion, CUDA_DEVICE, testloader_near_OOD, noiseMagnitude1, temper, confidences_baseline_Near_OOD, confidences_ODIN_Near_OOD)
    
    print("Processing far out-of-distribution images")
    get_score(net1, criterion, CUDA_DEVICE, testloader_far_OOD, noiseMagnitude1, temper, confidences_baseline_Far_OOD, confidences_ODIN_Far_OOD)




