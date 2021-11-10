from plugin_collection import FrontendPlugin
import torch
import torch.nn as nn
import numpy as np
import json

class Pytorch(FrontendPlugin):
    """Pytorch frontend plugin transforms given Pytorch pt-file to the intermediate format"""

    #? Mapping dictionary for layer name mapping
    layerMappings = { 'Linear' : 'Dense',
                    'Conv1d':'Conv1D',
                    'Conv2d':'Conv2D',
                    'MaxPool2d':'MaxPooling2D',
                    'MaxPool1d':'MaxPooling1D',
                    'AvgPool2d':'AveragePooling2D',
                    'AvgPool1d':'AveragePooling1D',
                    'ReLU':'ReLU',
                    'Sigmoid':'Sigmoid',
                    'Tanh':'Tanh',
                    'Softmax':'Softmax'
                    }

    def __init__(self):
        super().__init__('pytorch', 'Pytorch Frontend Plugin')

    def transform_to_intermediate_format(self, input):
        """Returns the intermediate format represenation of the given pt-file"""
        output = { "class_name":"Sequential", "config":{"name":"sequential_1", "layers":[]}}

        #? Loading the given pytorch model
        model = torch.load(input)

        counter=0
        #? Iterating over model layers and extracting relevant information for conversion
        for layer in model:
            out_layer = dict()
            out_layer["class_name"] = self.layerMappings[str(type(layer).__name__)]
            out_layer["config"] = dict()
            #? Processing activation layer specific information
            if (type(layer)==torch.nn.modules.activation.ReLU or type(layer)==torch.nn.modules.activation.Sigmoid
                or type(layer)==torch.nn.modules.activation.Tanh or type(layer)==torch.nn.modules.activation.Softmax):
                output["config"]["layers"][counter-1]["config"]["activation"] = str(type(layer).__name__).lower()
                out_layer = None
                counter = counter -1
            #? Processing dense layer specific information
            elif (type(layer)==torch.nn.modules.linear.Linear):
                if (counter==0):
                    out_layer["config"]["batch_input_shape"] = [None, layer.in_features]
                out_layer["config"]["units"]= layer.out_features
                out_layer["kernel_values"] = layer.weight.detach().numpy().tolist()
                if (layer.bias is not None):
                    out_layer["bias_values"] = layer.bias.detach().numpy().tolist()
                    out_layer["config"]["use_bias"] = True
                else:
                    out_layer["config"]["use_bias"] = False
                out_layer["config"]["activation"]="linear"
            #? Processing pooling layer specific information
            elif (type(layer)==torch.nn.modules.pooling.MaxPool2d or type(layer)==torch.nn.modules.pooling.MaxPool1d
                or type(layer)==torch.nn.modules.pooling.AvgPool2d or type(layer)==torch.nn.modules.pooling.AvgPool1d):
                if (counter==0):
                    out_layer["config"]["batch_input_shape"] = [None, layer.in_channels]
                out_layer["config"]["pool_size"] = list(tuple(layer.kernel_size))
                out_layer["config"]["strides"] = list(tuple(layer.stride))

                # TODO: Calculate padding size here and add it in the next line

                out_layer["config"]["padding"] = [0,0,0,0] if layer.padding==0 else "same"
                if (type(layer)==torch.nn.modules.pooling.MaxPool2d or type(layer)==torch.nn.modules.pooling.MaxPool1d):
                    out_layer["config"]["dilation"] = [layer.dilation]

            if (out_layer is not None):
                output["config"]["layers"].append(out_layer)
            counter=counter+1

        #? Returning a json representation of the built model
        return json.loads(json.dumps(output))