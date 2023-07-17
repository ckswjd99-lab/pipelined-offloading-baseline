from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from .utils import configHeadTail

RESNET50_SPLIT_IDX = 20
RESNET50_INTER_SIZE = 100490
RESNET50_RESULT_SIZE = 4129

original_model = ResNet50(weights='imagenet')

class ResNet50Head:
    def __init__(self, split_point=RESNET50_SPLIT_IDX):
        self.split_point = split_point
        self.layers = original_model.layers[:self.split_point]
    
    def __call__(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

class ResNet50Tail:
    def __init__(self, split_point=RESNET50_SPLIT_IDX):
        self.split_point = split_point
        self.layers = original_model.layers[self.split_point:]
    
    def __call__(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

inputPreprocessor = preprocess_input
outputDecoder = decode_predictions

if __name__ == "__main__":
    configHeadTail(ResNet50Head, ResNet50Tail)