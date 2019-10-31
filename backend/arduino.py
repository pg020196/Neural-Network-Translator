from plugin_collection import BackendPlugin
from backend import backend_utils
from shutil import copyfile
import os

class Arduino(BackendPlugin):
    """Arduino backend plugin translates the intermediate format to native Arduino code"""

    #? Dictionaries for mapping activation functions and layer types to integer values
    activation_functions = {'linear':0,'sigmoid':1, 'relu':2, 'tanh':3, 'softmax':4}
    layer_types = {'dense':0, 'flatten':1,
                   'maxpooling1d':2, 'maxpooling2d':3, 'maxpooling3d':4,
                   'avgpooling1d':5, 'avgpooling2d':6, 'avgpooling3d':7,
                   'conv1d':8, 'conv2d':9,'conv3d':10}
    padding_types = {'valid':0, 'same':1}

    def __init__(self):
        super().__init__('arduino','Arduino Backend Plugin', None)

    def translate_to_native_code(self, input, outputfile):
        """Translates the given input (intermediate format) to native Arduino code and writes a header- and a ino-file"""

        markers = self.build_markers(input)

        #? Reading the header file with markers and replacing them with the markers array
        header_file = backend_utils.replace_markers(backend_utils.read_marker_file('./backend/nn_model.h-template'), markers)

        #? Creating directory if not existing
        out_directory_path = '_out/' + os.path.splitext(outputfile)[0]
        if not os.path.exists(out_directory_path):
            os.makedirs(out_directory_path)

        header_filename = out_directory_path + '/nn_model.h'
        with open(header_filename, 'w') as file:
            file.write(header_file)

        c_file_source_path = './backend/nn_model.c-template'
        c_file_destination_path = out_directory_path + '/nn_model.c'
        ino_file_source_path = './backend/arduino.ino-template'
        ino_file_destination_path = out_directory_path + '/' + os.path.splitext(outputfile)[0] + '.ino'

        #? Copying files in defined output directory
        copyfile(c_file_source_path, c_file_destination_path)
        copyfile(ino_file_source_path, ino_file_destination_path)

    def build_markers(self, input):
        """Returns a markers array build from intermediate input information """
        markers = dict()

        #? common markers
        markers['###numberLayers###'] = backend_utils.get_number_of_layers(input)
        markers['###dimNumberLayers###'] = markers['###numberLayers###'] - 1
        markers['###layerTypes###'] = backend_utils.get_layer_types_string(input, self.layer_types)

        layerOutputHeight, layerOutputWidth = backend_utils.get_output_dimensions(input)
        markers['###layerOutputWidth###'] = backend_utils.convert_array_to_string(layerOutputWidth)
        markers['###layerOutputHeight###'] = backend_utils.convert_array_to_string(layerOutputHeight)

        #? Dense layer specific markers
        markers['###activationFunctions###'] = backend_utils.get_activation_function_string(input, self.activation_functions)

        weight_indices_string, weights_array = backend_utils.get_weight_information(input, layerOutputHeight)
        markers['###weights###'] = backend_utils.convert_array_to_string(weights_array)
        markers['###dimWeights###'] = len(weights_array)
        markers['###indicesWeights###'] = weight_indices_string

        use_bias_string, bias_indices_string, bias_array = backend_utils.get_bias_information(input)
        markers['###bias###'] = backend_utils.convert_array_to_string(bias_array)
        markers['###dimBias###'] = len(bias_array)
        markers['###indicesBias###'] = bias_indices_string
        markers['###useBias###'] = use_bias_string

        #? Pooling layer specific markers
        poolHeights, poolWidths = backend_utils.get_pool_size_strings(input)
        markers['###poolWidth###'] = poolWidths
        markers['###poolHeight###'] = poolHeights

        verticalStrides, horizontalStrides = backend_utils.get_strides_strings(input)
        markers['###horizontalStride###'] = horizontalStrides
        markers['###verticalStride###'] = verticalStrides
        markers['###padding###'] = backend_utils.get_padding_string(input, self.padding_types)

        return markers