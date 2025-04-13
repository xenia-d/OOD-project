import numpy as np
import torch
from torch.nn import functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ProgressBar
from DUQ.DUQ import CNN_DUQ  
from Data import FashionMNIST, EMNIST, MNIST  
from DUQ.OOD import get_auroc_ood, get_anomaly_targets_and_scores
from Utils.utils import plot_roc_curve, get_best_model_idx

import matplotlib.pyplot as plt

def train_model(l_gradient_penalty, length_scale, final_model, id_train, id_val, near_ood_val, far_ood_val, device):
    mnist_train_loader, mnist_val_loader = id_train, id_val

    fashion_val_loader = far_ood_val
    emnist_val_loader = near_ood_val

    num_classes = 10
    embedding_size = 256
    learnable_length_scale = False 
    gamma = 0.999

    model = CNN_DUQ(
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def output_transform_bce(output):
        y_pred, y = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y = output
        return y_pred, torch.argmax(y, dim=1)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        y = F.one_hot(y, num_classes=10).float()

        x, y = x.to(device), y.to(device)  

        y_pred = model(x)
        loss = F.binary_cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        x, y = batch
        y = F.one_hot(y, num_classes=10).float()
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
            evaluator.run(mnist_val_loader)
            _, roc_auc_fashionmnist = get_auroc_ood(mnist_val_loader, fashion_val_loader, model, device)
            _, roc_auc_emnist = get_auroc_ood(mnist_val_loader, emnist_val_loader, model, device)

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

    evaluator.run(mnist_val_loader)
    mnist_test_accuracy = evaluator.state.metrics["accuracy"]

    return model, mnist_test_accuracy  



def main():
    # Load id, near ood & far ood
    mnist = MNIST(batch_size=64)
    mnist_test_loader = mnist.get_test()

    fashionmnist = FashionMNIST(batch_size=64)
    _, fashion_val_loader = fashionmnist.get_train()
    fashion_test_loader = fashionmnist.get_test()

    emnist = EMNIST(batch_size=64)
    _, emnist_val_loader = emnist.get_train()
    emnist_test_loader = emnist.get_test()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # For mac

    l_gradient_penalties = [0.0]
    length_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    repetition = 5
    final_model = False

    results = {}

    model_ood_aurocs = []
    all_models = []
    # Loop through hyperparameters and find best val ood detection (auroc)
    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:
            mnist_val_accuracies = []
            val_roc_aucs_fashionmnist = []
            val_roc_aucs_emnist = []
            models = []
            for _ in range(repetition):
                print(" ### NEW MODEL ### ")
                mnist_train_loader, mnist_val_loader = mnist.get_train()
                model, mnist_val_accuracy = train_model( 
                    l_gradient_penalty, length_scale, final_model, mnist_train_loader, mnist_val_loader, emnist_val_loader, fashion_val_loader, device
                )
                print("Model trained")
                val_roc_auc_fashionmnist = get_auroc_ood(mnist_val_loader, fashion_val_loader, model, device)
                val_roc_auc_emnist = get_auroc_ood(mnist_val_loader, emnist_val_loader, model, device)
                print("AUROC FashionMNIST:", val_roc_auc_fashionmnist, "AUROC EMNIST:", val_roc_auc_emnist, "Val Accuracy:", mnist_val_accuracy)

                mnist_val_accuracies.append(mnist_val_accuracy)
                val_roc_aucs_fashionmnist.append(val_roc_auc_fashionmnist)
                val_roc_aucs_emnist.append(val_roc_auc_emnist)

                models.append(model) 


            results[f"lgp{l_gradient_penalty}_ls{length_scale}"] = [
                (np.mean(mnist_val_accuracies), np.std(mnist_val_accuracies)),
                (np.mean(val_roc_aucs_fashionmnist), np.std(val_roc_aucs_fashionmnist)),
                (np.mean(val_roc_aucs_emnist), np.std(val_roc_aucs_emnist)),
            ]
            print(results[f"lgp{l_gradient_penalty}_ls{length_scale}"])
            model_ood_aurocs.append((np.mean(val_roc_aucs_fashionmnist), np.mean(val_roc_aucs_emnist)))
            all_models.append(models)

    # Test on model with best validation ood detection
    best_model_on_val_idx = get_best_model_idx(model_ood_aurocs)
    best_sigma = length_scales[best_model_on_val_idx % len(length_scales)]
    best_models = all_models[best_model_on_val_idx] 

    for i, best_model in enumerate(best_models): 
        print("Evaluating best model OOD detection on test set, sigma:", best_sigma)

        id_scores, ood_scores = get_anomaly_targets_and_scores(best_model, mnist_test_loader, fashion_test_loader, device)
        plot_roc_curve(id_scores, ood_scores, "DUQ -- MNIST vs FashionMNIST (" + str(best_sigma) + ") " + str(i), method="DUQ", dist="MNIST")

        id_scores, ood_scores = get_anomaly_targets_and_scores(best_model, mnist_test_loader, emnist_test_loader, device)
        plot_roc_curve(id_scores, ood_scores, "DUQ -- MNIST vs EMNIST (" + str(best_sigma) + ") " + str(i), method="DUQ", dist="MNIST")

        torch.save(best_model.state_dict(), "Saved Models/baseline_duq_" + str(i) + ".pth")

    print(results)

main()

