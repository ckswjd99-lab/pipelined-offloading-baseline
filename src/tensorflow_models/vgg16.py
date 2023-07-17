from tensorflow.keras.applications.vgg16 import VGG16
from .utils import configHeadTail

VGG16_SPLIT_IDX = 20
VGG16_INTER_SIZE = 100490
VGG16_RESULT_SIZE = 4129

original_model = VGG16(weights='imagenet')

class Vgg16Head:
    def __init__(self, split_point=VGG16_SPLIT_IDX):
        self.split_point = split_point
        self.layers = original_model.layers[:self.split_point]
    
    def __call__(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

class Vgg16Tail:
    def __init__(self, split_point=VGG16_SPLIT_IDX):
        self.split_point = split_point
        self.layers = original_model.layers[self.split_point:]
    
    def __call__(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

if __name__ == "__main__":
    configHeadTail(Vgg16Head, Vgg16Tail)