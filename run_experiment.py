import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from Data.MNIST import MNIST
from Data.EMNIST import EMNIST
from Data.FashionMNIST import FashionMNIST
from Model_Architecture.Baseline_CNN import BaselineCNN
from Utils.utils import *

def load_model(model_name):
    model = BaselineCNN()
    model.load_state_dict(torch.load("Saved_Models/"+model_name))
    model.eval()
    return model

def get_baseline_entropys(model, data_loader):
    all_entropys = []
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            softmax_outputs = torch.exp(outputs) # Convert from log softmax to softmax
            # print("Outputs: ", softmax_outputs)
            entropys = scipy.stats.entropy(softmax_outputs, axis=1)
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

def get_ensemble_entropys(models, data_loader):
    all_entropys = []
    with torch.no_grad():
        for images, labels in data_loader:
            all_outputs = []
            for model in models:
                outputs = model(images)
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

def main():
    models = []
    for num in range(1,6):
        model = load_model("baseline_cnn_"+str(num)+".pth")
        models.append(model)

    # Get MNIST Entropys
    mnist_data = MNIST(batch_size=64)
    mnist_test = mnist_data.get_test()
    mnist_entropys_baseline = get_baseline_entropys(models[0], mnist_test)
    mnist_entropys_ensemble = get_ensemble_entropys(models, mnist_test)

    # Get EMNIST Entropys
    emnist_data = EMNIST(batch_size=64)
    emnist_test = emnist_data.get_test()
    emnist_entropys_baseline = get_baseline_entropys(models[0], emnist_test)
    emnist_entropys_ensemble = get_ensemble_entropys(models, emnist_test)

    plot_roc_curve(mnist_entropys_baseline, emnist_entropys_baseline, "Baseline -- MNIST vs EMNIST")
    plot_roc_curve(mnist_entropys_ensemble, emnist_entropys_ensemble, "Ensemble -- MNIST vs EMNIST")

    # Get FashionMNIST Entropys
    fashion_mnist_data = FashionMNIST(batch_size=64)
    fashion_mnist_test = fashion_mnist_data.get_test()
    fashion_mnist_entropys_baseline = get_baseline_entropys(models[0],fashion_mnist_test)
    fashion_mnist_entropys_ensemble = get_ensemble_entropys(models, fashion_mnist_test)

    plot_roc_curve(mnist_entropys_baseline, fashion_mnist_entropys_baseline, "MNIST vs Fashion MNIST")
    plot_roc_curve(mnist_entropys_ensemble, fashion_mnist_entropys_ensemble, "Ensemble -- MNIST vs Fashion MNIST")

    # Plot Overall Histograms
    all_dataset_entropys_baseline = [mnist_entropys_baseline, emnist_entropys_baseline, fashion_mnist_entropys_baseline]
    plot_hist(all_dataset_entropys_baseline, "Baseline -- Histogram of Dataset Entropys")

    all_dataset_entropys_ensemble = [mnist_entropys_ensemble, emnist_entropys_ensemble, fashion_mnist_entropys_ensemble]
    plot_hist(all_dataset_entropys_ensemble, "Ensemble -- Histogram of Dataset Entropys")

main()