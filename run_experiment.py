import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from Data.MNIST import MNIST
from Data.EMNIST import EMNIST
from Data.FashionMNIST import FashionMNIST
from Data.CIFAR10 import CIFAR10
from Data.CIFAR100 import CIFAR100
from Data.SVHN import SVHN
from Model_Architecture.Baseline_CNN import BaselineCNN
from Model_Architecture.Adv_CNN import ResNet18
from Utils.utils import plot_hist, plot_roc_curve, plot_density

def load_model(model_name, device, dataset="MNIST"):
    if dataset == "MNIST":
        model = BaselineCNN()
    elif dataset == "CIFAR10":
        model = ResNet18()
    
    model.load_state_dict(torch.load("Saved_Models/"+model_name))
    model.eval()
    model.to(device)

    return model

def get_baseline_entropys(model, data_loader, device):
    all_entropys = []
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)          
            outputs = model(images)
            softmax_outputs = torch.exp(outputs) # Convert from log softmax to softmax
            # print("Outputs: ", softmax_outputs)
            entropys = scipy.stats.entropy(softmax_outputs.cpu(), axis=1)
            # print("Entropys:", entropys)
            for idx, entropy in enumerate(entropys):
                all_entropys.append(entropy)
                # if entropy > 0.75:
                #     plt.imshow(images[idx].squeeze())
                #     plt.title("High Entropy: " + str(labels[idx].item()))
                #     plt.show()
                # if entropy < 0.05:
                #     plt.imshow(images[idx].squeeze())
                #     plt.title("Low Entropy: " + str(labels[idx].item()))
                #     plt.show()
    print(str(len(all_entropys)), "Baseline Entropys Obtained")

    return all_entropys

def get_ensemble_entropys(models, data_loader, device):
    all_entropys = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            all_outputs = []
            for model in models:
                outputs = model(images).cpu()
                softmax_outputs = torch.exp(outputs) # Convert from log softmax to softmax
                all_outputs.append(softmax_outputs)
            # print("All Outputs: ", all_outputs)
            mean_outputs = np.mean(all_outputs, axis=0)
            # print("Mean Output: ", mean_outputs)
            entropys = scipy.stats.entropy(mean_outputs, axis=1)
            for idx, entropy in enumerate(entropys):
                all_entropys.append(entropy)
                # if entropy > 0.75:
                #     plt.imshow(images[idx].squeeze())
                #     plt.title("High Entropy: " + str(labels[idx].item()))
                #     plt.show()
                # if entropy < 0.05:
                #     plt.imshow(images[idx].squeeze())
                #     plt.title("Low Entropy: " + str(labels[idx].item()))
                #     plt.show()
            # break
    print(str(len(all_entropys)), "Ensemble Entropys Obtained")

    return all_entropys

def basic_experiment():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # For mac
    models = []
    for num in range(1,6):
        model = load_model("baseline_cnn_"+str(num)+".pth", device)
        models.append(model)

    # Get MNIST Entropys (ID)
    mnist_data = MNIST(batch_size=64)
    mnist_test = mnist_data.get_test()
    mnist_entropys_baseline = get_baseline_entropys(models[0], mnist_test, device)
    mnist_entropys_ensemble = get_ensemble_entropys(models, mnist_test, device)

    # Get EMNIST Entropys (Near OOD)
    emnist_data = EMNIST(batch_size=64)
    emnist_test = emnist_data.get_test()
    emnist_entropys_baseline = get_baseline_entropys(models[0], emnist_test, device)
    emnist_entropys_ensemble = get_ensemble_entropys(models, emnist_test, device)

    plot_roc_curve(mnist_entropys_baseline, emnist_entropys_baseline, "Baseline -- MNIST vs EMNIST")
    plot_roc_curve(mnist_entropys_ensemble, emnist_entropys_ensemble, "Ensemble -- MNIST vs EMNIST")

    # Get FashionMNIST Entropys (Far OOD)
    fashion_mnist_data = FashionMNIST(batch_size=64)
    fashion_mnist_test = fashion_mnist_data.get_test()
    fashion_mnist_entropys_baseline = get_baseline_entropys(models[0],fashion_mnist_test, device)
    fashion_mnist_entropys_ensemble = get_ensemble_entropys(models, fashion_mnist_test, device)

    plot_roc_curve(mnist_entropys_baseline, fashion_mnist_entropys_baseline, "MNIST vs Fashion MNIST")
    plot_roc_curve(mnist_entropys_ensemble, fashion_mnist_entropys_ensemble, "Ensemble -- MNIST vs Fashion MNIST")

    # Plot Overall Histograms
    all_dataset_entropys_baseline = [mnist_entropys_baseline, emnist_entropys_baseline, fashion_mnist_entropys_baseline]
    plot_hist(all_dataset_entropys_baseline, "Baseline -- Histogram of Dataset Entropys", bins=10)
    plot_density(all_dataset_entropys_baseline, "Baseline -- Density Plot of Dataset Entropys")

    all_dataset_entropys_ensemble = [mnist_entropys_ensemble, emnist_entropys_ensemble, fashion_mnist_entropys_ensemble]
    plot_hist(all_dataset_entropys_ensemble, "Ensemble -- Histogram of Dataset Entropys", bins=10)
    plot_density(all_dataset_entropys_ensemble, "Ensemble -- Density Plot of Dataset Entropys")

def adv_experiment():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # For mac
    models = []
    for num in range(1,4):
        model = load_model("adv_cnn_"+str(num)+".pth", device, dataset="CIFAR10")
        models.append(model)

    # Get CIFAR10 Entropys (ID)
    cifar10_data = CIFAR10(batch_size=64)
    cifar10_test = cifar10_data.get_test()
    cifar10_entropy_baseline = get_baseline_entropys(models[0], cifar10_test, device)
    # cifar10_entropy_ensemble = get_ensemble_entropys(models, cifar10_test, device)

    # Get CIFAR100 Entropys (Near OOD)
    cifar100_data = CIFAR100(batch_size=64)
    cifar100_test = cifar100_data.get_test()
    cifar100_entropy_baseline = get_baseline_entropys(models[0], cifar100_test, device)
    # cifar100_entropy_ensemble = get_ensemble_entropys(models, cifar100_test, device)

    plot_roc_curve(cifar10_entropy_baseline, cifar100_entropy_baseline, "Baseline -- CIFAR10 (ID) vs CIFAR100 (Near OOD)")
    # plot_roc_curve(cifar10_entropy_ensemble, cifar100_entropy_ensemble, "Ensemble -- CIFAR10 (ID) vs CIFAR100 (Near OOD)")

    # Get SVHN Entropys (Far OOD)
    svhn_data = SVHN(batch_size=64)
    svhn_test = svhn_data.get_test()
    svhn_entropys_baseline = get_baseline_entropys(models[0], svhn_test, device)
    # svhn_entropy_ensemble = get_ensemble_entropys(models, svhn_test, device)

    plot_roc_curve(cifar10_entropy_baseline, svhn_entropys_baseline, "Baseline -- CIFAR10 (ID) vs SVHN (Far OOD)")
    # plot_roc_curve(cifar10_entropy_ensemble, svhn_entropy_ensemble, "Ensemble -- CIFAR10 (ID) vs SVHN (Far OOD)")

    # Plot overall metrics
    all_dataset_entropys_baseline = [cifar10_entropy_baseline, cifar100_entropy_baseline, svhn_entropys_baseline]
    # all_dataset_entropys_ensemble = [cifar10_entropy_ensemble, cifar100_entropy_ensemble, svhn_entropy_ensemble]

    plot_hist(all_dataset_entropys_baseline, "Baseline -- Histogram of Dataset Entropys", bins=10, legend=["CIFAR10", "CIFAR100", "SVHN"])
    plot_density(all_dataset_entropys_baseline, "Baseline -- Density Plot of Dataset Entropys", legend=["CIFAR10", "CIFAR100", "SVHN"])

    # plot_hist(all_dataset_entropys_ensemble, "Ensemble -- Histogram of Dataset Entropys", bins=10, legend=["CIFAR10", "CIFAR100", "SVHN"])
    # plot_density(all_dataset_entropys_ensemble, "Ensemble -- Density Plot of Dataset Entropys", legend=["CIFAR10", "CIFAR100", "SVHN"])

# basic_experiment()
adv_experiment()