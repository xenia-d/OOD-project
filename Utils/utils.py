import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
from sklearn.metrics import roc_auc_score, roc_curve

def plot_roc_curve(id_uq, ood_uq, title, method, dist):
    labels = np.concatenate([np.zeros_like(id_uq), np.ones_like(ood_uq)])
    preds = np.concatenate([id_uq, ood_uq])
    fpr, tpr, thresholds = roc_curve(labels, preds)
    # Save fpr and tpr to a file
    os.makedirs("Saved Rocks/"+str(method)+"/"+str(dist), exist_ok=True)
    np.savez("Saved Rocks/"+str(method)+"/"+str(dist)+"/"+str(title), fpr=fpr, tpr=tpr)

    auroc = roc_auc_score(labels, preds)
    print("\t", title, "AUROC: ", round(auroc, 3))

    plt.plot(fpr, tpr)
    full_title = title + " -- AUROC = " + str(round(auroc, 3))
    plt.title(full_title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    os.makedirs("Saved Plots/"+str(method)+"/"+str(dist), exist_ok=True)
    plt.savefig("Saved Plots/"+str(method)+"/"+str(dist)+"/ROC "+str(title)+".png")
    plt.show()
    plt.close()

    return auroc

def plot_hist(data, title, method, dist, legend=["MNIST", "EMNIST", "Fashion MNIST"], bins=10):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.legend(legend)
    os.makedirs("Saved Plots/"+str(method)+"/"+str(dist), exist_ok=True)
    plt.savefig("Saved Plots/"+str(method)+"/"+str(dist)+"/Histogram "+str(title))
    # plt.show()
    plt.close()

def plot_density(data, method, dist, title = "Density Plot of Entropys", legend=["MNIST", "EMNIST", "Fashion MNIST"]):
    plt.figure(figsize=(8, 5))
    for i, values in enumerate(data):
        sn.kdeplot(values, label=legend[i], fill=True, alpha=0.5)

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    os.makedirs("Saved Plots/"+str(method)+"/"+str(dist), exist_ok=True)
    plt.savefig("Saved Plots/"+str(method)+"/"+str(dist)+"/Density "+str(title))
    # plt.show()
    plt.close()

def show_max_entropy_examples(image, label):
    plt.imshow(image.squeeze())
    plt.title("High Entropy: " + str(label.item()))
    # plt.show()

def get_best_model_idx(auroc_list):
    avg_aurocs = []
    for far_ood_auroc, near_ood_auroc in auroc_list:
        avg_aurocs.append((far_ood_auroc + near_ood_auroc) / 2)
    best_model_idx = np.argmax(avg_aurocs)
    return best_model_idx



