import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

class CIFAR10:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )

    def get_train_val(self):
        train_data = datasets.CIFAR10(root='Data', train=True, download=True, transform=self.transform)
        train_data, val_data = random_split(train_data, [0.9, 0.1])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader
    
    def get_test(self):
        test_data = datasets.CIFAR10(root='Data', train=False, download=True, transform=self.transform)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return test_loader