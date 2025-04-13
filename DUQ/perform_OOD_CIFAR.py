import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import torch.nn as nn
from torchvision.models import resnet18
from OOD import get_anomaly_targets_and_scores
from Data import SVHN, CIFAR10, CIFAR100
from Utils.utils import plot_roc_curve, get_best_model_idx
from DUQ import ResNet_DUQ

model_output_size = 512
feature_extractor = resnet18()

feature_extractor.conv1 = torch.nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
feature_extractor.maxpool = torch.nn.Identity()
feature_extractor.fc = torch.nn.Identity()


model = ResNet_DUQ(
    feature_extractor,
    num_classes=10,
    centroid_size=model_output_size, 
    model_output_size=model_output_size,
    length_scale=0.1, 
    gamma=0.999, 
)


device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")


model = model.to(device)
model.load_state_dict(torch.load("Saved Models/advanced_duq.pt", map_location=device))
model.eval()

cifar10 = CIFAR10(batch_size=64)
cifar10_test_loader = cifar10.get_test()

cifar100 = CIFAR100(batch_size=64)
cifar100_test_loader = cifar100.get_test()

SVHN = SVHN(batch_size=64)
svhn_test_loader = SVHN.get_test()


id_scores, ood_scores = get_anomaly_targets_and_scores(model, cifar10_test_loader, svhn_test_loader, device)
plot_roc_curve(id_scores, ood_scores, "DUQ -- CIFAR10 vs SVHN (", method="DUQ", dist="CIFAR10")

id_scores, ood_scores = get_anomaly_targets_and_scores(model, cifar10_test_loader, cifar100_test_loader, device)
plot_roc_curve(id_scores, ood_scores, "DUQ -- CIFAR10 vs CIFAR100 (", method="DUQ", dist="CIFAR10")


