import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

class FashionMNIST:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])


    def get_train(self):
        train_data = datasets.FashionMNIST(root='Data', train=True, download=True, transform=self.transform)
        train_data, val_data = random_split(train_data, [0.9, 0.1])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader
    
    def get_test(self):
        test_data = datasets.FashionMNIST(root='Data', train=False, download=True, transform=self.transform)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return test_loader
    
    def get_name(self):
        return "Fashion MNIST"