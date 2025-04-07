import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_auc_score, roc_curve


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

def plot_hist(data, title, legend=["MNIST", "EMNIST", "Fashion MNIST"], bins=10):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.legend(legend)
    plt.show()

def plot_density(data, title = "Density Plot of Entropys", legend=["MNIST", "EMNIST", "Fashion MNIST"]):
    plt.figure(figsize=(8, 5))
    for i, values in enumerate(data):
        sn.kdeplot(values, label=legend[i], fill=True, alpha=0.5)

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()



