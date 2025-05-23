import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import argparse

from Model_Architecture.baseline_cnn import BaselineCNN
from Data.preprocess import inspect_class_distribution


def train_model(model, train_data, device, epochs, batch_size, lr):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_i % 64 == 63:  # Print loss every 64 batches
                print(f'[{epoch + 1}, {batch_i + 1}] loss: {running_loss / 64:.4f}')
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
    f1 = f1_score(all_labels, all_preds, average='weighted')  

    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'F1-Score on test set: {f1:.4f}')

    return accuracy, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST Model')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs to train')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--model_num', default=1, type=int,
                        help='model number to save')
    
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    model_num = args.model_num
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN()
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='Data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='Data', train=False, download=True, transform=transform)

    inspect_class_distribution(train_data, "Training Data Distribution")
    inspect_class_distribution(test_data, "Test Data Distribution")

    train_model(model, train_data, device, epochs, batch_size, lr)
    test_model(model, test_data, device)

    torch.save(model.state_dict(), "Saved_Models/baseline_cnn_" + str(model_num) + ".pth")
