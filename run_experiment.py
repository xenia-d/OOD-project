import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from Data.MNIST import MNIST
from Data.EMNIST import EMNIST
from Data.FashionMNIST import FashionMNIST
from Model_Architecture.Baseline_CNN import BaselineCNN

def load_model():
    model = BaselineCNN()
    model.load_state_dict(torch.load("Saved_Models/baseline_cnn.pth"))
    return model

def plot_roc_curve(id_uq, ood_uq, title):
    labels = np.concatenate([np.zeros_like(id_uq), np.ones_like(ood_uq)])
    preds = np.concatenate([id_uq, ood_uq])
    fpr, tpr, thresholds = roc_curve(labels, preds)

    auroc = roc_auc_score(labels, preds)
    print("\t", title, "AUROC: ", round(auroc, 3))

    plt.plot(fpr, tpr)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def get_entropys(model, data_loader):
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

    return all_entropys

def main():
    model = load_model()

    # Get MNIST Entropys
    mnist_data = MNIST(batch_size=64)
    mnist_test = mnist_data.get_test()
    mnist_entropys = get_entropys(model, mnist_test)
    print("~"+str(len(mnist_test)*64), "MMIST Entropys Obtained")

    # Get EMNIST Entropys
    emnist_data = EMNIST(batch_size=64)
    emnist_test = emnist_data.get_test()
    emnist_entropys = get_entropys(model, emnist_test)
    print("~"+str(len(emnist_test)*64), "EMMIST Entropys Obtained")
    plot_roc_curve(mnist_entropys, emnist_entropys, "MNIST vs EMNIST")

    # Get FashionMNIST Entropys
    fashion_mnist_data = FashionMNIST(batch_size=64)
    fashion_mnist_test = fashion_mnist_data.get_test()
    fashion_mnist_entropys = get_entropys(model,fashion_mnist_test)
    print("~"+str(len(fashion_mnist_test)*64), "Fashion MMIST Entropys Obtained")
    plot_roc_curve(mnist_entropys, fashion_mnist_entropys, "MNIST vs Fashion MNIST")

    all_dataset_entropys = [mnist_entropys, emnist_entropys, fashion_mnist_entropys]
    plt.hist(all_dataset_entropys, bins=8)
    plt.legend(["MNIST", "EMNIST", "Fashion MNIST"])
    plt.show()

main()