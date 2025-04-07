import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

class CIFAR100:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )

    def get_train(self):
        train_data = datasets.CIFAR100(root='Data', train=True, download=True, transform=self.transform)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        return train_loader
    
    def get_test(self):
        test_data = datasets.CIFAR100(root='Data', train=False, download=True, transform=self.transform)
        filtered_test_data = self.filter_classes_out(test_data)
        test_loader = DataLoader(filtered_test_data, batch_size=self.batch_size, shuffle=True)

        return test_loader
    
    def filter_classes_out(self, data):
        filter_in = [3, 4, 8, 13, 15, 21, 31, 34, 36, 38, 48, 63, 64, 65, 66, 69, 74, 75, 80, 88, 89, 90, 97] # bear, beaver, bicycle, bus, camel, chimp, elephant, 
         # fox, hamster, kangaroo, motorcycle, porcupine, possum, rabbit, raccoon, rocket, shrew, skunk, squirrel, tiger, tractor, train, wolf
        filtered_indicies = []
        for idx, data_point in enumerate(data):
            _, label = data_point
            if label in filter_in:
                filtered_indicies.append(idx)

        filtered_data = Subset(data, filtered_indicies)

        return filtered_data
    
    def show_sample(self, data):
        images, labels = next(iter(data))
        for image, label in zip(images, labels):
            image = np.transpose(image, (1,2,0))
            plt.imshow(image)
            plt.title(str(label.item()))
            plt.show()
