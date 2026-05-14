import torch.nn as nn
from torchvision import models

class DenseNet14(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNet14, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)
