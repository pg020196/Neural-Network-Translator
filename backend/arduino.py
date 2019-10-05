from plugin_collection import BackendPlugin
import json
from itertools import chain

class Arduino(BackendPlugin):

    activation_functions = {'linear':0,'sigmoid':1, 'relu':2, 'tanh':3, 'softmax':4}

    def __init__(self):
        super().__init__('arduino','Arduino Backend Plugin', None)

    def replace_markers(self, file, markers):
        for marker, value in markers.items():
            file = file.replace(marker,str(value))
        return file

    def read_marker_file(self, filename):
        with open(filename, 'r') as file:
            return file.read()

    def build_activation_function_string(self, input):
        string = '{'
        for layer in input['config']['layers']:
            string = string + str(self.activation_functions[layer['config']['activation'].lower()]) + ','
        return string[:-1] + '}'

    def build_use_bias_string(self, input):
        string = '{'
        for layer in input['config']['layers']:
            string = string + str(layer['config']['use_bias']).lower() + ','
        return string[:-1] + '}'

    def build_units_in_layers_string(self, input):
        first_layer = True
        string = '{'
        for layer in input['config']['layers']:
            if (first_layer):
                string = string + str(layer['config']['batch_input_shape'][1]) + ','
                first_layer=False
            string = string + str(layer['config']['units']) + ','
        return string[:-1] + '}'

    def get_number_of_layers(self, input):
        count=0
        for layer in input['config']['layers']:
            count = count + 1
        return count + 1

    def get_bias_array(self, input, numberLayers):
        count=0
        output= [None] * numberLayers
        for layer in input['config']['layers']:
            output[count] = layer['bias_values']
            count = count + 1
        return list(chain.from_iterable(output))

    def get_weights_array(self, input, numberLayers):
        count=0
        output= [None] * numberLayers
        for layer in input['config']['layers']:
            output[count] = list(chain.from_iterable(layer['kernel_values']))
            count = count + 1
        return list(chain.from_iterable(output))

    def build_indices_weights_string(self, input):
        last_layer_values = 0
        string = '{'
        for layer in input['config']['layers']:
            string = string + str(last_layer_values) + ','
            last_layer_values = last_layer_values + int(layer['config']['units']) * int(layer['config']['batch_input_shape'][1])
        return string[:-1] + '}'

    def build_indices_bias_string(self, input):
        last_layer_values = 0
        string = '{'
        for layer in input['config']['layers']:
            string = string + str(last_layer_values) + ','
            last_layer_values = last_layer_values + layer['config']['units']
        return string[:-1] + '}'

    def convert_array_to_string(self, array):
        string = '{'
        for value in array:
            string = string + str(value) + ','
        return string[:-1] + '}'

    def translate_to_native_code(self, input, outputfile):
        file = self.read_marker_file('./backend/arduino_marker_file.ino')

        markers = dict()

        markers['###input_data###'] = '{6, 148, 72, 35, 0, 33.6, 0.627, 50}'

        markers['###numberLayers###'] = self.get_number_of_layers(input)
        markers['###dimNumberLayers###'] = markers['###numberLayers###'] - 1
        markers['###activationFunctions###'] = self.build_activation_function_string(input)
        markers['###useBias###'] = self.build_use_bias_string(input)
        markers['###unitsInLayers###'] = self.build_units_in_layers_string(input)
        weights_array = self.get_weights_array(input, markers['###dimNumberLayers###'])
        markers['###weights###'] = self.convert_array_to_string(weights_array)
        markers['###dimWeights###'] = len(weights_array)
        markers['###indicesWeights###'] = self.build_indices_weights_string(input)
        bias_array = self.get_bias_array(input, markers['###dimNumberLayers###'])
        markers['###bias###'] = self.convert_array_to_string(bias_array)
        markers['###dimBias###'] = len(bias_array)
        markers['###indicesBias###'] = self.build_indices_bias_string(input)

        native_code = self.replace_markers(file, markers)

        with open(outputfile, 'w') as file:
            file.write(native_code)