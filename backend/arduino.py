from plugin_collection import BackendPlugin
from backend import backend_utils
from shutil import copyfile
import os

class Arduino(BackendPlugin):
    """Arduino backend plugin translates the intermediate format to native Arduino code"""

    #? Dictionaries for mapping activation functions and layer types to integer values
    activation_functions = {'linear':0,'sigmoid':1, 'relu':2, 'tanh':3, 'softmax':4}
    layer_types = {'dense':0}

    def __init__(self):
        super().__init__('arduino','Arduino Backend Plugin', None)

    def translate_to_native_code(self, input, outputfile):
        """Translates the given input (intermediate format) to native Arduino code and writes a header- and a ino-file"""
        markers = dict()

        #? Building the markers array from the information in input
        markers['###numberLayers###'] = backend_utils.get_number_of_layers(input)
        markers['###dimNumberLayers###'] = markers['###numberLayers###'] - 1
        markers['###activationFunctions###'] = backend_utils.build_activation_function_string(input, self.activation_functions)
        markers['###useBias###'] = backend_utils.build_use_bias_string(input)
        markers['###unitsInLayers###'] = backend_utils.build_units_in_layers_string(input)
        weights_array = backend_utils.get_weights_array(input, markers['###dimNumberLayers###'])
        markers['###weights###'] = backend_utils.convert_array_to_string(weights_array)
        markers['###dimWeights###'] = len(weights_array)
        markers['###indicesWeights###'] = backend_utils.build_indices_weights_string(input)
        bias_array = backend_utils.get_bias_array(input, markers['###dimNumberLayers###'])
        markers['###bias###'] = backend_utils.convert_array_to_string(bias_array)
        markers['###dimBias###'] = len(bias_array)
        markers['###indicesBias###'] = backend_utils.build_indices_bias_string(input)
        markers['###layerTypes###'] = backend_utils.build_layer_types_string(input, self.layer_types)

        #? Reading the header file with markers and replacing them with the markers array
        header_file = backend_utils.replace_markers(backend_utils.read_marker_file('./backend/nn_model.h-template'), markers)

        out_directory_path = '_out/' + os.path.splitext(outputfile)[0]
        if not os.path.exists(out_directory_path):
            os.makedirs(out_directory_path)

        header_filename = out_directory_path + '/nn_model.h'
        with open(header_filename, 'w') as file:
            file.write(header_file)

        c_file_source_path = './backend/nn_model.c-template'
        c_file_destination_path = out_directory_path + '/nn_model.c'
        ino_file_source_path = './backend/arduino_main.ino'
        ino_file_destination_path = out_directory_path + '/' + os.path.splitext(outputfile)[0] + '.ino'

        copyfile(c_file_source_path, c_file_destination_path)
        copyfile(ino_file_source_path, ino_file_destination_path)