import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class SVHN:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )

    def get_train(self):
        train_data = datasets.SVHN(root='Data', split="train", download=True, transform=self.transform)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        return train_loader
    
    def get_test(self):
        test_data = datasets.SVHN(root='Data', split="test", download=True, transform=self.transform)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return test_loader