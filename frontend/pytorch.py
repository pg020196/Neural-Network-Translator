from plugin_collection import FrontendPlugin
import torch
import torch.nn as nn
import numpy as np
import json

class Pytorch(FrontendPlugin):
    """Pytorch frontend plugin transforms given Pytorch pt-file to the intermediate format"""

    def __init__(self):
        super().__init__('pytorch', 'Pytorch Frontend Plugin')

    def transform_to_intermediate_format(self, input):
        """Returns the intermediate format represenation of the given pt-file"""
        output = { "class_name":"Sequential", "config":{"name":"sequential_1", "layers":[]}}

        model = torch.load(input)

        counter=0
        for layer in model:
            out_layer = dict()
            out_layer["class_name"] = str(type(layer).__name__)
            out_layer["config"] = dict()
            if (type(layer)==torch.nn.modules.activation.ReLU or type(layer)==torch.nn.modules.activation.Sigmoid
                or type(layer)==torch.nn.modules.activation.Tanh or type(layer)==torch.nn.modules.activation.Softmax):
                output["config"]["layers"][counter-1]["config"]["activation"] = str(type(layer).__name__).lower()
                out_layer = None
                counter = counter -1
            elif (type(layer)==torch.nn.modules.linear.Linear):
                if (counter==0):
                    out_layer["config"]["batch_input_shape"] = [None, layer.in_features]
                out_layer["config"]["units"]= layer.out_features
                out_layer["kernel_values"] = layer.weight.detach().numpy().tolist()
                if (layer.bias is not None):
                    out_layer["bias_values"] = layer.bias.detach().numpy().tolist()
                out_layer["config"]["activation"]="linear"
            elif (type(layer)==torch.nn.modules.conv.Conv2d or type(layer)==torch.nn.modules.conv.Conv1d):
                if (counter==0):
                    out_layer["config"]["batch_input_shape"] = [None, layer.in_channels]
                out_layer["config"]["kernel_size"] = list(tuple(layer.kernel_size))
                out_layer["config"]["strides"] = list(tuple(layer.stride))
                out_layer["config"]["padding"] = "valid" if layer.padding==0 else "same"
                out_layer["config"]["dilation"] = list(tuple(layer.dilation))
            elif (type(layer)==torch.nn.modules.pooling.MaxPool2d or type(layer)==torch.nn.modules.pooling.MaxPool1d
                or type(layer)==torch.nn.modules.pooling.AvgPool2d or type(layer)==torch.nn.modules.pooling.AvgPool1d):
                if (counter==0):
                    out_layer["config"]["batch_input_shape"] = [None, layer.in_channels]
                out_layer["config"]["kernel_size"] = list(tuple(layer.kernel_size))
                out_layer["config"]["strides"] = list(tuple(layer.stride))
                out_layer["config"]["padding"] = "valid" if layer.padding==0 else "same"
                if (type(layer)==torch.nn.modules.pooling.MaxPool2d or type(layer)==torch.nn.modules.pooling.MaxPool1d):
                    out_layer["config"]["dilation"] = [layer.dilation]

            if (out_layer is not None):
                output["config"]["layers"].append(out_layer)
            counter=counter+1

        return json.loads(json.dumps(output))