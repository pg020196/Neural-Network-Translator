from plugin_collection import BackendPlugin
import json
import os
from itertools import chain


class Arduino(BackendPlugin):
    """Arduino backend plugin translates the intermediate format to native Arduino code"""

    #? Dictionaries for mapping activation functions and layer types to integer values
    activation_functions = {'linear':0,'sigmoid':1, 'relu':2, 'tanh':3, 'softmax':4}
    layer_types = {'dense':0}

    def __init__(self):
        super().__init__('arduino','Arduino Backend Plugin', None)

    def replace_markers(self, file, markers):
        """ Replaces all given markes in given file with their respective value"""
        for marker, value in markers.items():
            file = file.replace(marker,str(value))
        return file

    def read_marker_file(self, filename):
        """ Reads given filename"""
        with open(filename, 'r') as file:
            return file.read()

    def build_activation_function_string(self, input):
        """Returns a string containing an array of indices representing the activation function for each layer"""
        string = '{'
        for layer in input['config']['layers']:
            #? Dictionary activation_functions contains the mapping to the indices
            string = string + str(self.activation_functions[layer['config']['activation'].lower()]) + ','
        return string[:-1] + '}'

    def build_use_bias_string(self, input):
        """Returns a string containing an array of bools indicating the usage of biases"""
        string = '{'
        for layer in input['config']['layers']:
            string = string + str(layer['config']['use_bias']).lower() + ','
        return string[:-1] + '}'

    def build_units_in_layers_string(self, input):
        """Returns a string containing an array of number of units for each layer"""
        first_layer = True
        string = '{'
        for layer in input['config']['layers']:
            #? First layer has to be treated different because of input shape
            if (first_layer):
                string = string + str(layer['config']['batch_input_shape'][1]) + ','
                first_layer=False
            string = string + str(layer['config']['units']) + ','
        return string[:-1] + '}'

    def build_layer_types_string(self, input):
        """Returns a string containing an array of indices representing the layer type for each layer"""
        string = '{'
        for layer in input['config']['layers']:
            #? Dictionary layer_types contains the mapping to the indices
            string = string + str(self.layer_types[layer['class_name'].lower()]) + ','
        return string[:-1] + '}'

    def build_indices_weights_string(self, input):
        """Returns a string containing an array of indices indicating the start position of weights for each layer"""
        last_layer_values = 0
        string = '{'
        for layer in input['config']['layers']:
            string = string + str(last_layer_values) + ','
            last_layer_values = last_layer_values + int(layer['config']['units']) * int(layer['config']['batch_input_shape'][1])
        return string[:-1] + '}'

    def build_indices_bias_string(self, input):
        """Returns a string containing an array of indices indicating the start position of biases for each layer"""
        last_layer_values = 0
        string = '{'
        for layer in input['config']['layers']:
            string = string + str(last_layer_values) + ','
            last_layer_values = last_layer_values + layer['config']['units']
        return string[:-1] + '}'

    def convert_array_to_string(self, array):
        """Returns a string containing the given array"""
        string = '{'
        for value in array:
            string = string + str(value) + ','
        return string[:-1] + '}'

    def get_number_of_layers(self, input):
        """Returns the number of layers in the neural network"""
        count=0
        for layer in input['config']['layers']:
            count = count + 1
        return count + 1

    def get_bias_array(self, input, numberLayers):
        """Returns a flattened array of bias values"""
        count=0
        output= [None] * numberLayers
        for layer in input['config']['layers']:
            output[count] = layer['bias_values']
            count = count + 1
        #? Flattening the array before returning
        return list(chain.from_iterable(output))

    def get_weights_array(self, input, numberLayers):
        """Returns a flattened array of weights values"""
        count=0
        output= [None] * numberLayers
        for layer in input['config']['layers']:
            output[count] = list(chain.from_iterable(layer['kernel_values']))
            count = count + 1
        #? Flattening the array before returning
        return list(chain.from_iterable(output))

    def translate_to_native_code(self, input, outputfile):
        """Translates the given input (intermediate format) to native Arduino code and writes a header- and a ino-file"""
        markers = dict()

        #? Building the markers array from the information in input
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
        markers['###layerTypes###'] = self.build_layer_types_string(input)

        #? Reading the header file with markers and replacing them with the markers array
        header_file = self.replace_markers(self.read_marker_file('./backend/arduino_config_marker.h'), markers)

        header_filename = os.path.splitext(outputfile)[0] + '.h'
        with open(header_filename, 'w') as file:
            file.write(header_file)

        #? Reading the ino file with markers and inserting the correct header filename for the import
        ino_file = self.replace_markers(self.read_marker_file('./backend/arduino_marker.ino'), {'###headerfile###': header_filename})

        with open(os.path.splitext(outputfile)[0] + '.ino', 'w') as file:
            file.write(ino_file)