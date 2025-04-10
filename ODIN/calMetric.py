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
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import misc
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sn
from Utils.utils import plot_roc_curve, plot_hist, plot_density

def log_results(log_file, experiment_name, id_uq, near_ood_uq, far_ood_uq):
    labels = np.concatenate([np.zeros_like(near_ood_uq), np.ones_like(id_uq)])
    preds = np.concatenate([near_ood_uq, id_uq])
    auroc = roc_auc_score(labels, preds)
    log_file.write(f"{experiment_name} -- Near OOD AUROC: {round(auroc, 3)}\n")

    labels = np.concatenate([np.zeros_like(far_ood_uq), np.ones_like(id_uq)])
    preds = np.concatenate([far_ood_uq, id_uq])
    auroc = roc_auc_score(labels, preds)
    log_file.write(f"{experiment_name} -- Far OOD AUROC: {round(auroc, 3)}\n")
    log_file.write("\n")

def new_metric(nn, num):
    print("Starting Metric Calculation")
    # [in, near, far]
    if nn == "ADVANCED_CNN":
        folder_name = 'adv_cnn'
        name_list = ["CIFAR10", "CIFAR100", "SVH"]
        near_dataset = "CIFAR100"
        far_dataset = "SVHN"
        dist = "CIFAR10"
    elif nn == "BASELINE_CNN":
        folder_name = 'baseline_cnn'
        name_list = ["MNIST", "EMNIST", "FashionMNIST"]
        near_dataset = "EMNIST"
        far_dataset = "FashionMNIST"
        dist = "MNIST"

    # open a file to log results
    log_file = open(f"ODIN/results/{folder_name}/{num}_Log.txt", "w")

    ## BASELINE ##
    experiment_name = "Conf -- " + dist + " (ID) vs "
    iid = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/{num}_confidence_Base_In.txt', delimiter=',')
    near_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/{num}_confidence_Base_Near_Out.txt', delimiter=',')
    far_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/{num}_confidence_Base_Far_Out.txt', delimiter=',')
    id_uq = iid[:, 2]
    near_ood_uq = near_ood[:, 2]
    far_ood_uq = far_ood[:, 2]

    log_results(log_file, nn + " Conf " + num, id_uq, near_ood_uq, far_ood_uq)
    plot_roc_curve(near_ood_uq, id_uq, experiment_name+near_dataset+" (Near OOD) "+num, "Conf", dist)
    plot_roc_curve(far_ood_uq, id_uq, experiment_name+far_dataset+" (Far OOD) "+num, "Conf", dist)
    plot_hist(data=[id_uq, near_ood_uq, far_ood_uq], title=f"Conf {dist} -- Histogram of Dataset Confidences {num}", legend = name_list, bins=10, dist=dist, method="ODIN")
    plot_density(data=[id_uq, near_ood_uq, far_ood_uq], title=f"Conf {dist} -- Density Plot of Dataset Confidences {num}", legend=name_list, dist=dist, method="ODIN")

    ## ODIN ##
    experiment_name = "ODIN -- " + dist + " (ID) vs "

    iid = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/{num}_confidence_ODIN_In.txt', delimiter=',')
    near_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/{num}_confidence_ODIN_Near_Out.txt', delimiter=',')
    far_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/{num}_confidence_ODIN_Far_Out.txt', delimiter=',')
    id_uq = iid[:, 2]
    near_ood_uq = near_ood[:, 2]
    far_ood_uq = far_ood[:, 2]
    
    log_results(log_file, nn + " ODIN " + num, id_uq, near_ood_uq, far_ood_uq)
    plot_roc_curve(near_ood_uq, id_uq, experiment_name+near_dataset+" (Near OOD) " + num, "ODIN", dist)
    plot_roc_curve(far_ood_uq, id_uq, experiment_name+far_dataset+" (Far OOD) " + num, "ODIN", dist)
    plot_hist(data=[id_uq, near_ood_uq, far_ood_uq], title=f"ODIN {dist} -- Histogram of Dataset Confidences {num}", legend=name_list, bins=10, dist=dist, method="ODIN")
    plot_density(data=[id_uq, near_ood_uq, far_ood_uq], title=f"ODIN {dist} -- Density Plot of Dataset Confidences {num}", legend=name_list, dist=dist, method="ODIN")



if __name__ == "__main__":
    new_metric("ADVANCED_CNN")







