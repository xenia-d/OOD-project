import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import time

from Model_Architecture.Adv_CNN import ResNet18
from Data.CIFAR10 import CIFAR10



def train_model(model, train_data, val_data, device, epochs, print_interval=500):    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_running_loss = 0.0
        val_running_loss = 0.0
        val_accuracy = 0.0
        interval_start = time.time()

        for batch_i, (inputs, labels) in enumerate(train_data, 0):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            if batch_i % print_interval == 0:  # Print loss every 'print_interval' batches
                time_taken = time.time() - interval_start
                print(f'[{epoch + 1}, {batch_i + 1}] train loss: {train_running_loss / print_interval:.4f} ({time_taken:.1f} seconds)')
                train_running_loss = 0.0
                interval_start = time.time()

        for inputs, labels in val_data: # Get val loss and accuracy
            model.eval()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            val_accuracy += correct / total
        val_accuracy /= len(val_data)
            
        print(f'\t val loss: {val_running_loss / print_interval:.4f}, val accuracy: {val_accuracy:.4f}')



    print("It finished training")

def test_model(model, test_dataset, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_dataset:
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # For mac

    for i in range(5):
        print("Training iteration: ", i+1)
        # Initialize the model
        model = ResNet18()
        model.to(device)

        cifar_10_data = CIFAR10(batch_size=64)
        cifar_10_train, cifar_10_val = cifar_10_data.get_train_val()
        cifar_10_test = cifar_10_data.get_test()

        train_model(model, cifar_10_train, cifar_10_val, device, epochs=20)
        test_model(model, cifar_10_test, device)

        torch.save(model.state_dict(), "Saved Models/adv_cnn_2_" + str(i+1) + ".pth")
