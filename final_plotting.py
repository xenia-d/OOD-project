import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d

def get_dataset_name(dataset, ood_type):
    if ood_type == "Near OOD" and dataset == "MNIST":
        other_dataset_name = "EMNIST"
    elif ood_type == "Far OOD" and dataset == "MNIST":
        other_dataset_name = "Fashion MNIST"
    elif ood_type == "Near OOD" and dataset == "CIFAR10":
        other_dataset_name = "CIFAR100"
    elif ood_type == "Far OOD" and dataset == "CIFAR10":
        other_dataset_name = "SVHN"

    return other_dataset_name

def load_roc(dataset, method, ood_type):
    """Load ROC curve data from a saved .npz file"""
    folder_path = f"Saved Rocks/{method}/{dataset}/"
    other_dataset_name = get_dataset_name(dataset, ood_type)
    file_name = f"{method} -- {dataset} (ID) vs {other_dataset_name} ({ood_type}).npz"
    data = np.load(os.path.join(folder_path, file_name))
    return data['fpr'], data['tpr']

def load_all_rocs(dataset):
    # Define methods and datasets to organize plots
    methods = ["Baseline", "Ensemble"]
    ood_types = ["Near OOD", "Far OOD"]
    
    rocs = []
    for method in methods:
        for ood_type in ood_types:
            fpr, tpr = load_roc(dataset, method, ood_type)
            rocs.append((fpr, tpr, dataset, method, ood_type))

    return rocs

def plot_all_rocs():
    datasets = ["MNIST", "CIFAR10"]
    colors = ['b', 'g', 'm', 'r']

    for dataset in datasets:
        rocs = load_all_rocs(dataset)
        title = 'ROC Curves for Different Methods on ' + dataset

        plt.figure(figsize=(10, 8))
        for i, (fpr, tpr, dataset, method, ood_type) in enumerate(rocs):
            if ood_type == "Near OOD":
                plt.plot(fpr, tpr, colors[i] + '-', label=f"{method} -- ({ood_type})")
            else:
                plt.plot(fpr, tpr, colors[i-1] + '--', label=f"{method} -- ({ood_type})")
        plt.plot([0, 1], [0, 1], 'k:')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.savefig("Saved Plots/Overall/"+title)
        plt.show()

                

plot_all_rocs()
