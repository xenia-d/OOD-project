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
# from Utils.utils import plot_roc_curve, plot_hist, plot_density

def plot_roc_curve(id_uq, ood_uq, title):
    labels = np.concatenate([np.zeros_like(id_uq), np.ones_like(ood_uq)])
    preds = np.concatenate([id_uq, ood_uq])
    fpr, tpr, thresholds = roc_curve(labels, preds)

    auroc = roc_auc_score(labels, preds)
    print("\t", title, "AUROC: ", round(auroc, 3))

    plt.plot(fpr, tpr)
    full_title = title + " -- AUROC = " + str(round(auroc, 3))
    plt.title(full_title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("Saved Plots/ODIN/ROC "+str(title)+".png")
    plt.show()

def plot_hist(data, title, legend=["MNIST", "Fashion MNIST"], bins=10):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.legend(legend)
    plt.savefig("Saved Plots/ODIN/Histogram "+str(title)+".png")
    plt.show()

def plot_density(data, title = "Density Plot of Entropys", legend=["MNIST", "Fashion MNIST"]):
    plt.figure(figsize=(8, 5))
    for i, values in enumerate(data):
        sn.kdeplot(values, label=legend[i], fill=True, alpha=0.5)

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("Saved Plots/ODIN/Density "+str(title)+".png")
    plt.show()


def metric(nn, data):
    print("Calculating metrics...")
    get_curve("BASELINE", 'Base')
    get_curve("ODIN", 'Our')

    if nn == "densenet10" or nn == "wideresnet10": indis = "CIFAR-10"
    if nn == "densenet100" or nn == "wideresnet100": indis = "CIFAR-100"
    if nn == "densenet10" or nn == "densenet100": nnStructure = "DenseNet-BC-100"
    if nn == "wideresnet10" or nn == "wideresnet100": nnStructure = "Wide-ResNet-28-10"
    if nn == "custom": indis = "MNIST"
    if nn == "custom": nnStructure = "Custom-CNN"
    
    if data == "Imagenet": dataName = "Tiny-ImageNet (crop)"
    if data == "Imagenet_resize": dataName = "Tiny-ImageNet (resize)"
    if data == "LSUN": dataName = "LSUN (crop)"
    if data == "LSUN_resize": dataName = "LSUN (resize)"
    if data == "iSUN": dataName = "iSUN"
    if data == "Gaussian": dataName = "Gaussian noise"
    if data == "Uniform": dataName = "Uniform Noise"
    if data == "FashionMNIST": dataName = "FashionMNIST"
    fprBase, fprNew = tpr95(indis)
    errorBase, errorNew = detection(indis)
    aurocBase, aurocNew = auroc(indis)
    auprinBase, auprinNew = auprIn(indis)
    auproutBase, auproutNew = auprOut(indis)
    print("{:31}{:>22}".format("Neural network architecture:", nnStructure))
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    print("")
    print("{:>34}{:>19}".format("Baseline", "Our Method"))
    print("{:20}{:13.1f}%{:>18.1f}% ".format("FPR at TPR 95%:",fprBase*100, fprNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("Detection error:",errorBase*100, errorNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUROC:",aurocBase*100, aurocNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR In:",auprinBase*100, auprinNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR Out:",auproutBase*100, auproutNew*100))

def new_metric(nn):
    # [in, near, far]
    if nn == "ADVANCED_CNN":
        folder_name = 'adv_cnn'
        name_list = ["CIFAR10", "CIFAR100", "SVH"]
    elif nn == "BASELINE_CNN":
        folder_name = 'baseline_cnn'
        name_list = ["MNIST", "EMNIST", "FashionMNIST"]
    ## BASELINE ##
    experiment_name = nn + " BASE"
    iid = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/confidence_Base_In.txt', delimiter=',')
    near_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/confidence_Base_Near_Out.txt', delimiter=',')
    far_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/confidence_Base_Far_Out.txt', delimiter=',')
    id_uq = iid[:, 2]
    near_ood_uq = near_ood[:, 2]
    far_ood_uq = far_ood[:, 2]

    plot_roc_curve(near_ood_uq, id_uq, experiment_name+" -- Near OOD")
    plot_roc_curve(far_ood_uq, id_uq, experiment_name+" -- Far OOD")
    plot_hist([id_uq, near_ood_uq, far_ood_uq], f"{experiment_name} -- Histogram of Dataset Confidences", legend = name_list, bins=10)
    plot_density([id_uq, near_ood_uq, far_ood_uq], f"{experiment_name} -- Density Plot of Dataset Confidences", legend=name_list)

    ## ODIN ##
    experiment_name = nn + " ODIN"

    iid = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/confidence_ODIN_In.txt', delimiter=',')
    near_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/confidence_ODIN_Near_Out.txt', delimiter=',')
    far_ood = np.loadtxt(f'ODIN/softmax_scores/{folder_name}/confidence_ODIN_Far_Out.txt', delimiter=',')
    id_uq = iid[:, 2]
    near_ood_uq = near_ood[:, 2]
    far_ood_uq = far_ood[:, 2]

    
    plot_roc_curve(near_ood_uq, id_uq, experiment_name+" -- Near OOD")
    plot_roc_curve(far_ood_uq, id_uq, experiment_name+" -- Far OOD")
    plot_hist([id_uq, near_ood_uq, far_ood_uq], f"{experiment_name} -- Histogram of Dataset Entropys", legend=name_list, bins=10)
    plot_density([id_uq, near_ood_uq, far_ood_uq], f"{experiment_name} -- Density Plot of Dataset Entropys", legend=name_list)



if __name__ == "__main__":
    new_metric("ADVANCED_CNN")







