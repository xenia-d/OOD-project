import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd
import re
import seaborn as sns

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

def load_roc(dataset, method, ood_type, iter_num=1):
    """Load ROC curve data from a saved .npz file"""
    folder_path = f"Saved Rocks/{method}/{dataset}/"
    other_dataset_name = get_dataset_name(dataset, ood_type)
    file_name = f"{method} -- {dataset} (ID) vs {other_dataset_name} ({ood_type}) {str(iter_num)}.npz"
    data = np.load(os.path.join(folder_path, file_name))
    return data['fpr'], data['tpr']

def load_all_rocs(dataset):
    # Define methods and datasets to organize plots
    methods = ["Baseline", "Ensemble", "ODIN", "Conf"]
    ood_types = ["Near OOD", "Far OOD"]
    
    rocs = []
    for method in methods:
        for ood_type in ood_types:
            fpr, tpr = load_roc(dataset, method, ood_type)
            rocs.append((fpr, tpr, dataset, method, ood_type))

    return rocs

def get_ood_dataset_name(filename):
    match = re.search(r" vs (.+?) \(", filename)
    ood_dataset = match.group(1) if match else print(filename, "NO OOD DATASET FOUND")
    if ood_dataset == "CIFAR100" or ood_dataset == "EMNIST":
        distance = "Near OOD"
    elif ood_dataset == "SVHN" or ood_dataset == "Fashion MNIST" or ood_dataset == "FashionMNIST":
        if ood_dataset == "Fashion MNIST":
            ood_dataset = "FashionMNIST"
        distance = "Far OOD"
    return ood_dataset, distance

def make_dataframe():
    # Create a DataFrame to store the results
    rows = []
    results_path = "./Saved Rocks"
    methods = os.listdir(results_path)
    for method in methods:
        method_path = os.path.join(results_path, method)
        datasets = os.listdir(method_path)
        for dataset in datasets:
            dataset_path = os.path.join(method_path, dataset)
            files = os.listdir(dataset_path)
            for file in files:
                if file.endswith(".npz"):
                    # print(file)
                    ood_dataset, distance = get_ood_dataset_name(file)
                    data = np.load(os.path.join(dataset_path, file))
                    fpr = data['fpr']
                    tpr = data['tpr']
                    roc_auc = np.trapz(tpr, fpr)
                    if method == "Baseline":
                        method = "Entropy"
                    if method == "Conf":
                        method = "Confidence"
                    # print(f"Method: {method}, ID Dataset: {dataset}, OOD Dataset: {ood_dataset}, AUROC: {roc_auc}, Distance: {distance}")
                    rows.append({
                        "Method": method,
                        "ID Dataset": dataset,
                        "OOD Dataset": ood_dataset,
                        "TPR": tpr,
                        "FPR": fpr,
                        "AUROC": roc_auc,
                        "Distance": distance
                    })
    df = pd.DataFrame(rows, columns=["Method", "ID Dataset", "OOD Dataset", "TPR", "FPR", "AUROC", "Distance"])
    # print(df[df["Method"] == "DUQ"])
    return df

def make_barplot(target_dataset=None):
    df = make_dataframe()
    # filter df to only include target_distance if specified
    if target_dataset:
        df = df[df["ID Dataset"] == target_dataset]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=df,
        x="Distance",
        y="AUROC",
        hue="Method",
        errorbar="se"
    )
    ax.set_title(f"AUROC for Different Methods on {target_dataset} Dataset")
    diff = df["AUROC"].max() - df["AUROC"].min()
    ax.set(ylim=(min(df["AUROC"]-(0.1*diff)), 1))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("Saved Plots/Overall/"+f"AUROC for Different Methods on {target_dataset} Dataset")
    plt.show()

def plot_all_rocs():
    colors = ['b', 'g', 'm', 'r', 'c', 'yellow']

    df = make_dataframe()

    for dataset in df["ID Dataset"].unique():
        plt.figure(figsize=(6, 5))
        title = "ROC Curves for Different Methods on " + dataset + " (ID)"
        plotted_combos = []
        for row in df[df["ID Dataset"] == dataset].iterrows():
            num, (method, _, _, tpr, fpr, _, ood_type) = row
            # Clean up method names for plotting
            if method == "Baseline":
                method = "Entropy"
            if method == "Conf":
                method = "Confidence"
            if ((method, ood_type)) not in plotted_combos: # only plot one per combo
                i = len(plotted_combos)
                if ood_type == "Near OOD":
                    plt.plot(fpr, tpr, colors[int(i/2)] + '-', label=f"{method} -- ({ood_type})")
                else:
                    plt.plot(fpr, tpr, colors[int(i/2)] + '--', label=f"{method} -- ({ood_type})")
                plotted_combos.append((method, ood_type))

        plt.plot([0, 1], [0, 1], 'k:')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.tight_layout()
        plt.legend()
        plt.savefig("Saved Plots/Overall/"+title)
        plt.show()

                

plot_all_rocs()
make_barplot("MNIST")
make_barplot("CIFAR10")