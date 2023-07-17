import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from .utils import modelSplitter, configHeadTail

RESNET50_SPLIT_IDX = -3
RESNET50_INTER_SIZE = 802959
RESNET50_RESULT_SIZE = 4134

weights = ResNet50_Weights.IMAGENET1K_V1
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
        self.tail = tail[:-1]
        self.classify = tail[-1]
    def forward(self, x):
        x = self.tail(x)
        x = torch.transpose(x, 3, 1)
        x = self.classify(x)
        
        return x

if __name__ == "__main__":
    configHeadTail(ResNet50Head, ResNet50Tail)
    