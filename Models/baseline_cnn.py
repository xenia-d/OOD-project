import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  

        print("Shape before flattening:", x.shape) 
        x = torch.flatten(x, start_dim=1)  
        print("Shape after flattening:", x.shape) 

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  

        return x
    
if __name__ == "__main__":
    model = BaselineCNN()
    print(model)

    # Example forward pass
    sample_input = torch.randn(1, 1, 28, 28)  # Simulated MNIST image
    output = model(sample_input)
    print("Output shape:", output.shape)  # Should be (1, 10)


