import argparse
import json
import pathlib
import random
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision.models import resnet18
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar
from DUQ.DUQ import ResNet_DUQ
from DUQ.OOD import get_cifar_svhn_ood, get_cifar_cifar100_ood, get_auroc_classification
from Data.CIFAR100 import CIFAR100
from Data.SVHN import SVHN
from Data.CIFAR10 import CIFAR10 


def main(
    architecture,
    batch_size,
    length_scale,
    centroid_size,
    learning_rate,
    l_gradient_penalty,
    gamma,
    weight_decay,
    final_model,
    output_dir,
):

    cifar10 = CIFAR10(batch_size=64)
    cifar_train_dataset, cifar_val_dataset = cifar10.get_train_val()
    cifar_test_dataset = cifar10.get_test()

    num_classes = 10

    # Split up training set
    idx = list(range(len(cifar_train_dataset)))
    random.shuffle(idx)

    # if final_model:
    #     train_dataset = cifar_train_dataset
    #     val_dataset = cifar_test_dataset
    # else:
    #     val_size = int(len(cifar_train_dataset) * 0.8)
    #     train_dataset = torch.utils.data.Subset(cifar_train_dataset, idx[:val_size])
    #     val_dataset = torch.utils.data.Subset(cifar_train_dataset, idx[val_size:])

    #     val_dataset.transform = (
    #         cifar_test_dataset.transform
    #     )  # Test time preprocessing for validation

    if architecture == "WRN":
        model_output_size = 640
        epochs = 200
        milestones = [60, 120, 160]
        feature_extractor = WideResNet()
    elif architecture == "ResNet18":
        model_output_size = 512
        epochs = 100
        milestones = [25, 50, 75]
        feature_extractor = resnet18()

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        feature_extractor.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        feature_extractor.maxpool = torch.nn.Identity()
        feature_extractor.fc = torch.nn.Identity()

    if centroid_size is None:
        centroid_size = model_output_size

    model = ResNet_DUQ(
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    def calc_gradients_input(x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

    def calc_gradient_penalty(x, y_pred):
        gradients = calc_gradients_input(x, y_pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def step(engine, batch):
        model.train()

        optimizer.zero_grad()

        x, y = batch
        x, y = x.to(device), y.to(device)

        x.requires_grad_(True)

        y_pred = model(x)

        y = F.one_hot(y, num_classes).float()

        loss = F.binary_cross_entropy(y_pred, y, reduction="mean")

        if l_gradient_penalty > 0:
            gp = calc_gradient_penalty(x, y_pred)
            loss += l_gradient_penalty * gp

        loss.backward()
        optimizer.step()

        x.requires_grad_(False)

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred = model(x)

        return {"x": x, "y": y, "y_pred": y_pred}

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")

    metric = Accuracy(output_transform=lambda out: (out["y_pred"], out["y"]))
    metric.attach(evaluator, "accuracy")

    def bce_output_transform(out):
        return (out["y_pred"], F.one_hot(out["y"], num_classes).float())

    metric = Loss(F.binary_cross_entropy, output_transform=bce_output_transform)
    metric.attach(evaluator, "bce")

    metric = Loss(
        calc_gradient_penalty, output_transform=lambda out: (out["x"], out["y_pred"])
    )
    metric.attach(evaluator, "gradient_penalty")

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    kwargs = {"num_workers": 0, "pin_memory": True}

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        loss = metrics["loss"]

        print(f"Train - Epoch: {trainer.state.epoch} Loss: {loss:.2f}")


        if trainer.state.epoch > (epochs - 5):
            accuracy, auroc = get_cifar_svhn_ood(model, device)
            print(f"Test Accuracy CIFAR10-SVHN: {accuracy}, AUROC: {auroc}")
            accuracy, auroc = get_cifar_cifar100_ood(model, device)
            print(f"Test Accuracy CIFAR10-CIFAR100: {accuracy}, AUROC: {auroc}")

            accuracy, auroc = get_auroc_classification(cifar_val_dataset, model, device)


        evaluator.run(cifar_val_dataset)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        bce = metrics["bce"]
        GP = metrics["gradient_penalty"]
        loss = bce + l_gradient_penalty * GP

        print(
            (
                f"Valid - Epoch: {trainer.state.epoch} "
                f"Acc: {acc:.4f} "
                f"Loss: {loss:.2f} "
                f"BCE: {bce:.2f} "
                f"GP: {GP:.2f} "
            )
        )


        scheduler.step()

    trainer.run(cifar_train_dataset, max_epochs=epochs)
    evaluator.run(cifar_test_dataset)
    acc = evaluator.state.metrics["accuracy"]

    print(f"Test - Accuracy {acc:.4f}")

    torch.save(model.state_dict(), f"{output_dir}/model.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    cifar100 = CIFAR100(batch_size=64)
    cifar100_train_dataset, cifar100_val_dataset = cifar100.get_train()
    cifar100_test_dataset = cifar100.get_test()

    svhn = SVHN(batch_size=64)
    svhn_train_dataset, svhn_val_dataset = svhn.get_train()
    svhn_test_dataset = svhn.get_test()

    parser.add_argument(
        "--architecture",
        default="ResNet18",
        choices=["ResNet18", "WRN"],
        help="Pick an architecture (default: ResNet18)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training (default: 128)",
    )

    parser.add_argument(
        "--centroid_size",
        type=int,
        default=None,
        help="Size to use for centroids (default: same as model output)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )

    parser.add_argument(
        "--l_gradient_penalty",
        type=float,
        default=0.75,
        help="Weight for gradient penalty (default: 0.75)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Decay factor for exponential average (default: 0.999)",
    )

    parser.add_argument(
        "--length_scale",
        type=float,
        default=0.1,
        help="Length scale of RBF kernel (default: 0.1)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay (default: 5e-4)"
    )

    parser.add_argument(
        "--output_dir", type=str, default="DUQ_CIFAR_Results", help="set output folder"
    )

    # Below setting cannot be used for model selection,
    # because the validation set equals the test set.
    parser.add_argument(
        "--final_model",
        action="store_true",
        default=False,
        help="Use entire training set for final model",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

    pathlib.Path(args.output_dir).mkdir(exist_ok=True)

    main(**kwargs)