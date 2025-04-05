import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

class EMNIST:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_train(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.EMNIST(root='Data', split='balanced', train=True, download=True, transform=transform)
        filtered_train_data = self.filter_numbers_out(train_data)
        train_loader = DataLoader(filtered_train_data, batch_size=self.batch_size, shuffle=True)

        return train_loader
    
    def get_test(self):
        transform = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.EMNIST(root='Data', split='balanced', train=False, download=True, transform=transform)
        filtered_test_data = self.filter_numbers_out(test_data)
        test_loader = DataLoader(filtered_test_data, batch_size=self.batch_size, shuffle=True)

        return test_loader
    
    def filter_numbers_out(self, data):
        filter_out = [0,1,2,3,4,5,6,7,8,9]
        filtered_indicies = []
        for idx, data_point in enumerate(data):
            _, label = data_point
            if label in filter_out:
                filtered_indicies.append(idx)

        filtered_data = Subset(data, filtered_indicies)

        return filtered_data
