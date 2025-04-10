import torch
import numpy as np
import scipy.stats
from PIL import Image

from Data.MNIST import MNIST
from Data.EMNIST import EMNIST
from Data.FashionMNIST import FashionMNIST
from Data.CIFAR10 import CIFAR10
from Data.CIFAR100 import CIFAR100
from Data.SVHN import SVHN
from Model_Architecture.baseline_cnn import BaselineCNN
from Model_Architecture.Adv_CNN import ResNet18
from Utils.utils import *

def load_model(model_name, device, dataset="MNIST"):
    if dataset == "MNIST":
        model = BaselineCNN()
    elif dataset == "CIFAR10":
        model = ResNet18()
    
    model.load_state_dict(torch.load("Saved Models/"+model_name))
    model.eval()
    model.to(device)

    return model

def get_models(version="baseline"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # For mac
    if version == "baseline":
        dataset = "MNIST"
        filename = "baseline_cnn_"
    else:
        dataset = "CIFAR10"
        filename = "adv_cnn_2_"
    models = []
    for num in range(1,6):
        model = load_model(filename+str(num)+".pth", device, dataset=dataset)
        models.append(model)

    return models, device

def get_basic_datasets(batch_size):
    # Get MNIST (ID)
    mnist_data = MNIST(batch_size=batch_size)

    # Get EMNIST (Near OOD)
    emnist_data = EMNIST(batch_size=batch_size)

    # Get FashionMNIST (Far OOD)
    fashion_mnist_data = FashionMNIST(batch_size=batch_size)

    return mnist_data, emnist_data, fashion_mnist_data

def get_adv_datasets(batch_size):
    # Get CIFAR10 Entropys (ID)
    cifar10_data = CIFAR10(batch_size=batch_size)

    # Get CIFAR100 Entropys (Near OOD)
    cifar100_data = CIFAR100(batch_size=batch_size)

    # Get SVHN Entropys (Far OOD)
    svhn_data = SVHN(batch_size=batch_size)

    return cifar10_data, cifar100_data, svhn_data

def get_baseline_entropys(models, data_loader, device, model_idx, method, show_examples=False):
    all_entropys = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            if method == "Ensemble":
                all_outputs = []
                for model in models:
                    outputs = model(images).cpu()
                    softmax_outputs = np.exp(outputs) # Convert from log softmax to softmax
                    all_outputs.append(softmax_outputs)
                final_output = np.mean(all_outputs, axis=0)
            else:
                model = models[model_idx]
                output = model(images)
                final_output = torch.exp(output).cpu() # Convert from log softmax to softmax

            entropys = scipy.stats.entropy(final_output, axis=1)
            for idx, entropy in enumerate(entropys):
                all_entropys.append(entropy)
                if show_examples and entropy > 0.75:
                    show_max_entropy_examples(images[idx], labels[idx])

    print(str(len(all_entropys)), "Baseline Entropys Obtained")

    return all_entropys

def experiment_entropy(models, device, id, near_ood, far_ood, method="Baseline", model_idx = 0):
    id_test, near_ood_test, far_ood_test = id.get_test(), near_ood.get_test(), far_ood.get_test()

    # Get Entropys (ID)
    id_entropys = get_baseline_entropys(models, id_test, device, model_idx, method)
    
    # Get Entropys (Near OOD)
    near_ood_entropys = get_baseline_entropys(models, near_ood_test, device, model_idx, method)
    near_aucroc = plot_roc_curve(id_entropys, near_ood_entropys, method+" -- " + id.get_name() + " (ID) vs " + near_ood.get_name() + " (Near OOD) " + str(model_idx+1), method=method, dist=id.get_name())

    # Get Entropys (Far OOD)
    far_ood_entropys = get_baseline_entropys(models, far_ood_test, device, model_idx, method)
    far_auroc = plot_roc_curve(id_entropys, far_ood_entropys, method+" -- " + id.get_name() + " (ID) vs " + far_ood.get_name() + " (Far OOD) " + str(model_idx+1), method=method, dist=id.get_name())

    # Plot Overall Histograms
    all_dataset_entropys_baseline = [id_entropys, near_ood_entropys, far_ood_entropys]
    plot_hist(all_dataset_entropys_baseline, method+" -- Histogram of " + id.get_name() + " Dataset Entropys", method, id.get_name(), bins=10)
    plot_density(all_dataset_entropys_baseline, method, id.get_name())

    return near_aucroc, far_auroc

def main():
    # Baseline (MNIST)
    models, device = get_models(version="baseline")
    mnist, emnist, fashion_mnist = get_basic_datasets(batch_size=64)

    for model_idx in range(5):
        near_auroc, far_auroc = experiment_entropy(models, device, mnist, emnist, fashion_mnist, method="Baseline", model_idx=model_idx)
    
    near_auroc, far_auroc = experiment_entropy(models, device, mnist, emnist, fashion_mnist, method="Ensemble")

    # Advanced (CIFAR10)
    models, device = get_models(version="adv")
    cifar10, cifar100, svhn = get_adv_datasets(batch_size=64)

    for model_idx in range(5):
        near_auroc, far_auroc = experiment_entropy(models, device, cifar10, cifar100, svhn, method="Baseline", model_idx=model_idx)
    
    near_auroc, far_auroc = experiment_entropy(models, device, cifar10, cifar100, svhn, method="Ensemble")

main()