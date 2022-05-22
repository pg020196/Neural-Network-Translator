﻿from plugin_collection import BackendPlugin
from backend import backend_utils
import os

class C_Sharp(BackendPlugin): 
    """C_Sharp backend plugin translates the intermediate format to native C#-code"""

    #? Dictionaries for mapping activation functions and layer types to integer values
    activation_functions = {'linear':0,'sigmoid':1, 'relu':2, 'tanh':3, 'softmax':4}
    layer_types = {'dense':1, 'flatten':2,
                   'maxpooling1d':3, 'maxpooling2d':3, 'maxpooling3d':3,
                   'averagepooling1d':4, 'averagepooling2d':4, 'averagepooling3d':4,
                   'conv1d':5, 'conv2d':5,'conv3d':5, 'dropout':0, 'activation':6}
    padding_types = {'valid':0, 'same':1}

    def __init__(self):
        super().__init__('c_sharp','C# Backend Plugin', None)

    def translate_to_native_code(self, input, outputfile, executable_file):
        """Translates the given input (intermediate format) to native C#-code and writes a cs-file"""

        markers = self.build_markers(input, outputfile)

        #? Reading the C# template file with markers and replacing them with the markers array
        # cs_file = backend_utils.replace_markers(backend_utils.read_marker_file('./backend/c_sharp/nn_model.cs-template'), markers)
        cs_file = backend_utils.replace_markers(backend_utils.read_marker_file('NeuralNetworkLib_CSharp/NeuralNetwork/NeuralNetwork/NeuralNetworkBuilder.cs-template'), markers)

        #? Creating directory if not existing
        # out_dir_path = '_out/' + os.path.splitext(outputfile)[0]
        out_dir_path = 'NeuralNetworkLib_CSharp/NeuralNetwork/NeuralNetwork/' + os.path.splitext(outputfile)[0]
        cs_file_name = f'{outputfile}.cs'

        backend_utils.write_cs_file(out_dir_path, cs_file, cs_file_name, executable_file)

    def build_markers(self, input, outname):
        """Returns a markers dict built from intermediate input information """
        markers = dict()

        #? common markers
        markers['###ClassName###'] = outname
        markers['###numberLayers###'] = backend_utils.get_number_of_layers(input)
        markers['###dimNumberLayers###'] = markers['###numberLayers###'] - 1
        markers['###layerTypes###'] = backend_utils.get_layer_types_string(input, self.layer_types)

        layerOutputHeight, layerOutputWidth, layerOutputDepth = backend_utils.get_output_dimensions_csharp_backend(input)
        markers['###layerOutputHeight###'] = backend_utils.convert_array_to_string(layerOutputHeight)
        markers['###layerOutputWidth###'] = backend_utils.convert_array_to_string(layerOutputWidth)
        markers['###layerOutputDepth###'] = backend_utils.convert_array_to_string(layerOutputDepth)

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
        poolHeights, poolWidths = backend_utils.get_pool_size_strings_c_sharp_backend(input)
        markers['###poolWidths###'] = poolWidths
        markers['###poolHeights###'] = poolHeights

        verticalStrides, horizontalStrides = backend_utils.get_strides_strings(input)
        markers['###verticalStride###'] = verticalStrides
        markers['###horizontalStride###'] = horizontalStrides
        markers['###padding###'] = backend_utils.get_padding_string(input, self.padding_types)

        return markers