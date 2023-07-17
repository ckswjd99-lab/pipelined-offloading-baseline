import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights
from .utils import modelSplitter, configHeadTail

ALEXNET_SPLIT_IDX = 2
ALEXNET_INTER_SIZE = 36991
ALEXNET_RESULT_SIZE = 4127

weights = AlexNet_Weights.IMAGENET1K_V1
original_model = models.alexnet(weights=weights)
head, tail = modelSplitter(original_model, ALEXNET_SPLIT_IDX)

class AlexNetHead(nn.Module):
    def __init__(self):
        super(AlexNetHead, self).__init__()
        self.head = head
    def forward(self, x):
        x = self.head(x)
        x = torch.flatten(x)
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