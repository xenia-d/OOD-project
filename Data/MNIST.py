import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class MNIST:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_train(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(root='Data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        return train_loader
    
    def get_test(self):
        transform = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.MNIST(root='Data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return test_loader