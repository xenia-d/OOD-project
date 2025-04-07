import torch
import numpy as np
import scipy.stats
from torch.utils.data import DataLoader
from Data.MNIST import MNIST
from Data.FashionMNIST import FashionMNIST
from Model_Architecture.Baseline_CNN import BaselineCNN
from Model_Architecture.DUQ import DUQ_BaselineCNN
from Utils.utils import plot_roc_curve, plot_hist, plot_density 

