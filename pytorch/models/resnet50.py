import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from .utils import modelSplitter, configHeadTail

RESNET50_SPLIT_IDX = -3

weights = ResNet50_Weights.DEFAULT
original_model = models.resnet50(weights=weights)
head, tail = modelSplitter(original_model, RESNET50_SPLIT_IDX)

class ResNet50Head(nn.Module):
    def __init__(self):
        super(ResNet50Head, self).__init__()
        self.head = head
    def forward(self, x):
        x = self.head(x)
        return x

class ResNet50Tail(nn.Module):
    def __init__(self):
        super(ResNet50Tail, self).__init__()
        self.tail = tail
    def forward(self, x):
        x = self.tail(x)
        return x

if __name__ == "__main__":
    configHeadTail(ResNet50Head, ResNet50Tail)