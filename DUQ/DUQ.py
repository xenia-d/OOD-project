from Model_Architecture.baseline_cnn import BaselineCNN_FeatureExtractor
import torch
from torch import nn
import torch.nn.functional as F

# Taken and modified from the DUQ repository: https://github.com/y0ast/deterministic-uncertainty-quantification

class CNN_DUQ(BaselineCNN_FeatureExtractor):
    def __init__(
        self,
        num_classes,
        embedding_size,          
        learnable_length_scale,  
        length_scale,            
        gamma,                   
    ):
        super().__init__()

        self.gamma = gamma
        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 128), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

    def last_layer(self, z):
        return torch.einsum("ij,mnj->imn", z, self.W)

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0) 

        # Compare z to each class embedding
        diff = z - embeddings.unsqueeze(0) 
        distances = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()

        return distances  

    def forward(self, x):
        z = self.feature_extractor(x)    
        z = self.last_layer(z)           
        y_pred = self.output_layer(z)    
        return y_pred

    def update_embeddings(self, x, y):
        z = self.last_layer(self.feature_extractor(x))  

        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        features_sum = torch.einsum("ijk,ik->jk", z, y) 
        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


class ResNet_DUQ(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma
        self.sigma = length_scale

        self.feature_extractor = feature_extractor

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

    def _extract_features(self, x):
        out = F.relu(self.feature_extractor.bn1(self.feature_extractor.conv1(x)))
        out = self.feature_extractor.layer1(out)
        out = self.feature_extractor.layer2(out)
        out = self.feature_extractor.layer3(out)
        out = self.feature_extractor.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)
        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self._extract_features(x)
        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        z = self._extract_features(x)
        y_pred = self.rbf(z)
        return y_pred
