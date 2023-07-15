import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights
from .utils import modelSplitter, configHeadTail

ALEXNET_SPLIT_IDX = -3

weights = AlexNet_Weights.DEFAULT
original_model = models.alexnet(weights=weights)
head, tail = modelSplitter(original_model, ALEXNET_SPLIT_IDX)

class AlexNetHead(nn.Module):
    def __init__(self):
        super(AlexNetHead, self).__init__()
        self.head = head
    def forward(self, x):
        x = self.head(x)
        return x

class AlexNetTail(nn.Module):
    def __init__(self):
        super(AlexNetTail, self).__init__()
        self.tail = tail
    def forward(self, x):
        x = self.tail(x)
        return x

if __name__ == "__main__":
    configHeadTail(AlexNetHead, AlexNetTail)