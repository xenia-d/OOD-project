import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Models.baseline_cnn import BaselineCNN
from sklearn.metrics import f1_score


def train_model(model, train_data, device, epochs):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print loss every 100 batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    print("It finished training")

def test_model(model, test_dataset, device):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")  # the F1-score is weighted to consider class imbalance

    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'F1-Score on test set: {f1:.4f}')

    return accuracy, f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN()
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='Data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='Data', train=False, download=True, transform=transform)

    train_model(model, train_data, device, epochs=5)
    test_model(model, test_data, device)


