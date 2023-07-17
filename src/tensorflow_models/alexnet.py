from tensorflow.keras.applications.alexnet import AlexNet
from .utils import configHeadTail

ALEXNET_SPLIT_IDX = 20
ALEXNET_INTER_SIZE = 36998
ALEXNET_RESULT_SIZE = 4129

original_model = AlexNet(weights='imagenet')

class AlexNet50Head:
    def __init__(self, split_point=ALEXNET_SPLIT_IDX):
        self.split_point = split_point
        self.layers = original_model.layers[:self.split_point]
    
    def __call__(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

class AlexNet50Tail:
    def __init__(self, split_point=ALEXNET_SPLIT_IDX):
        self.split_point = split_point
        self.layers = original_model.layers[self.split_point:]
    
    def __call__(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

if __name__ == "__main__":
    configHeadTail(AlexNet50Head, AlexNet50Tail)