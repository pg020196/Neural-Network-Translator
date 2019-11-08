from plugin_collection import BackendPlugin
import backend.gcc.backend_utils as backend_utils
import os

class GCC(BackendPlugin):
    """GCC backend plugin translates the intermediate format to native C-code"""

    #? Dictionaries for mapping activation functions and layer types to integer values
    activation_functions = {'linear':0,'sigmoid':1, 'relu':2, 'tanh':3, 'softmax':4}
    layer_types = {'dense':1, 'flatten':2,
                   'maxpooling1d':3, 'maxpooling2d':3, 'maxpooling3d':3,
                   'averagepooling1d':4, 'averagepooling2d':4, 'averagepooling3d':4,
                   'conv1d':5, 'conv2d':5,'conv3d':5, 'dropout':0, 'activation':6}
    padding_types = {'valid':0, 'same':1}

    def __init__(self):
        super().__init__('gcc','GCC Backend Plugin', None)

    def translate_to_native_code(self, input, outputfile, executable_file):
        """Translates the given input (intermediate format) to native C-code and writes a header- and a c-file"""

        markers = self.build_markers(input)

        #? Reading the header file with markers and replacing them with the markers array
        h_file = backend_utils.replace_markers(backend_utils.read_marker_file('./backend/gcc/nn_model.h-template'), markers)

        #? Creating directory if not existing
        out_dir_path = '_out/' + os.path.splitext(outputfile)[0]
        c_file_source_path = './backend/gcc/nn_model.c-template'
        c_file_name = 'nn_model.c'
        h_file_name = 'nn_model.h'

        backend_utils.write_header_and_c_file(out_dir_path, h_file, h_file_name, c_file_source_path, c_file_name, executable_file)

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