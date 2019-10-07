from plugin_collection import BackendPlugin
from backend import backend_utils
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
        header_file = backend_utils.replace_markers(backend_utils.read_marker_file('./backend/arduino_config_marker.h'), markers)

        header_filename = os.path.splitext(outputfile)[0] + '.h'
        with open(header_filename, 'w') as file:
            file.write(header_file)

        #? Reading the ino file with markers and inserting the correct header filename for the import
        ino_file = backend_utils.replace_markers(backend_utils.read_marker_file('./backend/arduino_marker.ino'), {'###headerfile###': header_filename})

        with open(os.path.splitext(outputfile)[0] + '.ino', 'w') as file:
            file.write(ino_file)