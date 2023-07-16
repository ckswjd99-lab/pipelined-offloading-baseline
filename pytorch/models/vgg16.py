import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
from .utils import modelSplitter, configHeadTail

VGG16_SPLIT_IDX = 2
VGG16_INTER_SIZE = 100490
VGG16_RESULT_SIZE = 4129

weights = VGG16_Weights.DEFAULT
original_model = models.vgg16(weights=weights)
head, tail = modelSplitter(original_model, VGG16_SPLIT_IDX)

class Vgg16Head(nn.Module):
    def __init__(self):
        super(Vgg16Head, self).__init__()
        self.head = head
    def forward(self, x):
        x = self.head(x)
        x = torch.flatten(x, 1)
        return x

class Vgg16Tail(nn.Module):
    def __init__(self):
        super(Vgg16Tail, self).__init__()
        self.tail = tail
    def forward(self, x):
        x = self.tail(x)
        return x

if __name__ == "__main__":
    configHeadTail(Vgg16Head, Vgg16Tail)