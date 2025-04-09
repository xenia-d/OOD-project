import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from Data import FashionMNIST, EMNIST, SVHN, CIFAR10, MNIST, CIFAR100

def prepare_ood_datasets(true_dataset, ood_dataset):
    true_dataset_object = true_dataset.dataset
    ood_dataset_object = ood_dataset.dataset
    ood_dataset_object.transform = true_dataset_object.transform

    datasets = [true_dataset_object, ood_dataset_object]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset_object)), torch.ones(len(ood_dataset_object)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=500, shuffle=False, num_workers=0, pin_memory=False
    )

    return dataloader, anomaly_targets

def loop_over_dataloader(model, dataloader):
    model.eval()

    scores = []
    accuracies = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data
            target = target

            output = model(data)
            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)  
            accuracies.append(accuracy.cpu().numpy())

            scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

def get_auroc_ood(true_dataset, ood_dataset, model):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies[: len(true_dataset)])  
    roc_auc = roc_auc_score(anomaly_targets, scores)  

    return accuracy, roc_auc

def get_auroc_classification(dataset, model):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=0, pin_memory=False
    )

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc

def get_mnist_fashionmnist_ood(model):
    mnist = MNIST(batch_size=64)
    fashionmnist = FashionMNIST(batch_size=64)
    mnist_test = mnist.get_test()  
    fashionmnist_test = fashionmnist.get_test()  

    # print("MNIST Test Size:", len(mnist_test.dataset))
    # print("FashionMNIST Test Size:", len(fashionmnist_test.dataset))

    return get_auroc_ood(mnist_test, fashionmnist_test, model)

def get_mnist_emnist_ood(model):
    mnist = MNIST(batch_size=2000)
    emnist = EMNIST(batch_size=2000)
    mnist_test = mnist.get_test()  
    emnist_test = emnist.get_test()  

    # print("MNIST Test Size:", len(mnist_test.dataset))
    # print("EMNIST Test Size:", len(emnist_test.dataset))

    return get_auroc_ood(mnist_test, emnist_test, model)

def get_cifar_svhn_ood(model):
    cifar10 = CIFAR10(batch_size=2000)
    svhn = SVHN(batch_size=2000)
    cifar10_test = cifar10.get_test()  
    svhn_test = svhn.get_test()  

    # print("CIFAR10 Test Size:", len(cifar10_test.dataset))
    # print("SVHN Test Size:", len(svhn_test.dataset))

    return get_auroc_ood(cifar10_test, svhn_test, model)

def get_cifar_cifar100_ood(model):
    cifar10 = CIFAR10(batch_size=2000)
    cifar100 = CIFAR100(batch_size=2000)
    cifar10_test = cifar10.get_test()  
    cifar100_test = cifar100.get_test()  

    # print("CIFAR10 Test Size:", len(cifar10_test.dataset))
    # print("CIFAR100 Test Size:", len(cifar100_test.dataset))

    return get_auroc_ood(cifar10_test, cifar100_test, model)

def get_anomaly_targets_and_scores(model, id_dataset, ood_dataset):
    id_scores, _ = loop_over_dataloader(model, id_dataset)
    ood_scores, _ = loop_over_dataloader(model, ood_dataset)

    return id_scores, ood_scores
