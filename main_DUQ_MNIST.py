import random
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ProgressBar
from DUQ.DUQ import CNN_DUQ  
from Data import FashionMNIST, EMNIST, MNIST  
from DUQ.OOD import get_mnist_fashionmnist_ood, get_mnist_emnist_ood, get_anomaly_targets_and_scores
from Utils.utils import plot_roc_curve
from matplotlib import pyplot as plt


def train_model(l_gradient_penalty, length_scale, final_model):
    mnist = MNIST(batch_size=64)
    mnist_train_loader = mnist.get_train()
    mnist_test_loader = mnist.get_test()

    fashionmnist = FashionMNIST(batch_size=64)
    fashion_test_loader = fashionmnist.get_test()

    emnist = EMNIST(batch_size=64)
    emnist_test_loader = emnist.get_test()

    num_classes = 10
    embedding_size = 256
    learnable_length_scale = True 
    gamma = 0.999

    model = CNN_DUQ(
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def output_transform_bce(output):
        y_pred, y = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y = output
        return y_pred, y #torch.argmax(y, dim=1)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x, y = x.to(device), y.to(device)  

        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)  # target should NOT be one-hot
        loss.backward()
        optimizer.step()

        return loss.item()


    def eval_step(engine, batch):
        model.eval()
        x, y = batch
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    Accuracy(output_transform=output_transform_acc).attach(evaluator, "accuracy")
    Loss(F.cross_entropy, output_transform=output_transform_bce).attach(evaluator, "loss")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.2)

    pbar = ProgressBar()
    pbar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        scheduler.step()

        if trainer.state.epoch % 5 == 0:
            evaluator.run(mnist_test_loader)
            _, roc_auc_fashionmnist = get_mnist_fashionmnist_ood(model)
            _, roc_auc_emnist = get_mnist_emnist_ood(model)

            metrics = evaluator.state.metrics

            print(
                f"Epoch: {trainer.state.epoch} "
                f"Acc: {metrics['accuracy']:.4f} "
                f"Loss: {metrics['loss']:.2f} "
                f"AUROC FashionMNIST: {roc_auc_fashionmnist:.2f} "
                f"AUROC EMNIST: {roc_auc_emnist:.2f} "
            )
            print(f"Sigma: {model.sigma}")

    trainer.run(mnist_train_loader, max_epochs=5)

    evaluator.run(mnist_test_loader)
    mnist_test_accuracy = evaluator.state.metrics["accuracy"]
    # mnist_test_accuracy = 0.0

    # evaluator.run(fashion_test_loader)
    # fashion_test_accuracy = evaluator.state.metrics["accuracy"]

    # evaluator.run(emnist_test_loader)
    # emnist_test_accuracy = None  # skip it

    # roc_auc_fashionmnist = get_mnist_fashionmnist_ood(model)
    # roc_auc_emnist = get_mnist_emnist_ood(model)

    return model, mnist_test_accuracy, model.sigma  



if __name__ == "__main__":
    mnist = MNIST(batch_size=64)
    mnist_test_loader = mnist.get_test()

    fashionmnist = FashionMNIST(batch_size=64)
    fashion_test_loader = fashionmnist.get_test()

    emnist = EMNIST(batch_size=64)
    emnist_test_loader = emnist.get_test()

    l_gradient_penalties = [0.0]
    length_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    repetition = 1
    final_model = False

    results = {}

    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:
            mnist_test_accuracies = []
            roc_aucs_fashionmnist = []
            roc_aucs_emnist = []

            for _ in range(repetition):
                print(" ### NEW MODEL ### ")
                model, mnist_accuracy, sigma = train_model( 
                    l_gradient_penalty, length_scale, final_model
                )
                print("Model trained")
                # roc_auc_fashionmnist = get_mnist_fashionmnist_ood(model)
                # roc_auc_emnist = get_mnist_emnist_ood(model)
                roc_auc_fashionmnist = (0.0, 0.0)  # Placeholder for AUROC FashionMNIST
                roc_auc_emnist = (0.0, 0.0)
                # print("AUROC FashionMNIST:", roc_auc_fashionmnist, "AUROC EMNIST:", roc_auc_emnist)

                id_scores, ood_scores = get_anomaly_targets_and_scores(model, mnist_test_loader, fashion_test_loader)
                plot_roc_curve(id_scores, ood_scores, "DUQ -- MNIST vs FashionMNIST (" + str(length_scale) + ")", method="DUQ", dist="MNIST")

                id_scores, ood_scores = get_anomaly_targets_and_scores(model, mnist_test_loader, emnist_test_loader)
                plot_roc_curve(id_scores, ood_scores, "DUQ -- MNIST vs EMNIST (" + str(length_scale) + ")", method="DUQ", dist="MNIST")

                mnist_test_accuracies.append(mnist_accuracy)
                roc_aucs_fashionmnist.append(roc_auc_fashionmnist)
                roc_aucs_emnist.append(roc_auc_emnist)


            results[f"lgp{l_gradient_penalty}_ls{length_scale}"] = [
                (np.mean(mnist_test_accuracies), np.std(mnist_test_accuracies)),
                (np.mean(roc_aucs_fashionmnist), np.std(roc_aucs_fashionmnist)),
                (np.mean(roc_aucs_emnist), np.std(roc_aucs_emnist)),
            ]
            print(results[f"lgp{l_gradient_penalty}_ls{length_scale}"])

    print(results)
