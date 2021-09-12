from itertools import chain
from shutil import copyfile
import os

#? Definition of the layer names/class names
DENSE_LAYER = 'Dense'
CONV_2D_LAYER = 'Conv2D'
CONV_1D_LAYER = 'Conv1D'
MAX_POOL_2D_LAYER = 'MaxPooling2D'
MAX_POOL_1D_LAYER = 'MaxPooling1D'
AVG_POOL_2D_LAYER = 'AveragePooling2D'
AVG_POOL_1D_LAYER = 'AveragePooling1D'
FLATTEN_LAYER = 'Flatten'
ACTIVATION_LAYER = 'Activation'

def replace_markers(file, markers):
    """ Replaces all given markes in given file with their respective value"""
    for marker, value in markers.items():
        file = file.replace(marker,str(value))
    return file

def read_marker_file(filename):
    """ Reads given filename"""
    with open(filename, 'r') as file:
        return file.read()

def write_header_and_c_file(out_dir, c_file, c_file_name, h_file_source, h_file_name, exec_file):
    """Writes header- and c-file in given output directory (created if necessary)"""
    #? Creating directory if not existing
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    c_filename = out_dir + '/' + c_file_name
    with open(c_filename, 'w') as file:
        file.write(c_file)

    h_file_dest = out_dir + '/' + h_file_name

    #? Copying files in defined output directory
    copyfile(h_file_source, h_file_dest)

    if (exec_file is not None):
        exec_file_dest = out_dir + '/' + os.path.basename(out_dir) + os.path.splitext(exec_file)[-1]
        copyfile(exec_file, exec_file_dest)

def convert_array_to_string(array):
    """Returns a string containing the given array"""
    string = '{'
    for value in array:
        string = string + str(value) + ','
    if (len(string)!=1):
        string = string[:-1]
    return string + '}'

def get_number_of_layers(input):
    """Returns the number of layers in the neural network"""
    count=0
    for layer in input['config']['layers']:
        count = count + 1
    return count + 1

def get_layer_types_string(input, layer_types):
    """Returns a string containing an array of indices representing the layer type for each layer"""
    array = []
    for layer in input['config']['layers']:
        #? Dictionary layer_types contains the mapping to the indices
        array.append(str(layer_types[layer['class_name'].lower()]))
    return convert_array_to_string(array)

def get_output_dimensions(input):
    """Returns an array with height values, an array with width values and an array with depth values of the given input"""
    height_array = []
    last_output_height=0
    width_array = []
    last_output_width=0
    depth_array = []
    last_output_depth =0

    for layer in input['config']['layers']:
        input_height = last_output_height
        input_width = last_output_width
        input_depth = last_output_depth

        if 'batch_input_shape' in layer['config']:
            #? Extraction of 1st dimension (height)
            height_array.append(layer['config']['batch_input_shape'][1])
            input_height = layer['config']['batch_input_shape'][1]

            last_output_height = input_height

            #? Extraction of 2nd dimension (width)
            if (len(layer['config']['batch_input_shape'])>2):
                width_array.append(layer['config']['batch_input_shape'][2])
                input_width = layer['config']['batch_input_shape'][2]

                last_output_width = input_width

                #? Extraction of 3rd dimension (depth)
                if (len(layer['config']['batch_input_shape'])>3):
                    depth_array.append(layer['config']['batch_input_shape'][3])
                    input_depth = layer['config']['batch_input_shape'][3]

                    last_output_depth = input_depth
                else:
                    depth_array.append(1)
                    input_depth = 1
                    last_output_depth = input_depth
            else:
                width_array.append(1)
                input_width = 1
                last_output_width = input_width
                depth_array.append(1)
                input_depth = 1
                last_output_depth = input_depth

        #? Differentiation between layer types and specific processing
        if (layer['class_name']==DENSE_LAYER):
            act_height = len(layer['kernel_values'][0])
            act_width = 1
            act_depth = 1
        if (layer['class_name']==FLATTEN_LAYER):
            act_height = last_output_height * last_output_width * last_output_depth
            act_width = 1
            act_depth = 1
        if (layer['class_name']==AVG_POOL_1D_LAYER or layer['class_name']==MAX_POOL_1D_LAYER):
            vertical_padding = 0
            #? Only calculating padding if it is enabled in the layer definition
            if (layer['config']['padding'].lower() == 'same'):
                vertical_padding = int((layer['config']['pool_size'][0] - 1) / 2)

            #? Calculating height in dependence of padding
            act_height = ((input_height - layer['config']['pool_size'][0] + 2 * vertical_padding) / layer['config']['strides'][0]) + 1
            act_width=1
            act_depth = last_output_depth

        if (layer['class_name']==AVG_POOL_2D_LAYER or layer['class_name']==MAX_POOL_2D_LAYER):
            vertical_padding = 0
            horizontal_padding = 0

            #? Only calculating padding if it is enabled in the layer definition
            if (layer['config']['padding'].lower() == 'same'):
                vertical_padding = int((layer['config']['pool_size'][0] - 1) / 2)
                horizontal_padding = int((layer['config']['pool_size'][1] - 1) / 2)

            #? Calculating height and width in dependence of padding
            act_height = ((input_height - layer['config']['pool_size'][0] + 2 * vertical_padding) / layer['config']['strides'][0]) + 1
            act_width = ((input_width - layer['config']['pool_size'][1] + 2 * horizontal_padding) / layer['config']['strides'][1]) + 1
            act_depth = last_output_depth

        height_array.append(int(act_height))
        width_array.append(int(act_width))
        depth_array.append(int(act_depth))

        last_output_height = act_height
        last_output_width = act_width
        last_output_depth = act_depth

    return height_array, width_array, depth_array

def get_pool_size_strings(input):
    """Returns an array with pool height values and an array with pool width values of the given input"""
    width_array=[]
    height_array=[]

    for layer in input['config']['layers']:
        if (layer['class_name']==MAX_POOL_2D_LAYER or layer['class_name']==AVG_POOL_2D_LAYER):
            height_array.append(layer['config']['pool_size'][0])
            width_array.append(layer['config']['pool_size'][1])
        elif (layer['class_name']==MAX_POOL_1D_LAYER or layer['class_name']==AVG_POOL_1D_LAYER):
            height_array.append(layer['config']['pool_size'][0])
            width_array.append(1)
        else:
            height_array.append(0)
            width_array.append(0)

    return convert_array_to_string(height_array), convert_array_to_string(width_array)

def get_strides_strings(input):
    """Returns an array with stride height values and an array with stride width values of the given input"""
    width_array=[]
    height_array=[]
    for layer in input['config']['layers']:
        if (layer['class_name']==MAX_POOL_2D_LAYER or layer['class_name']==AVG_POOL_2D_LAYER):
            height_array.append(layer['config']['strides'][0])
            width_array.append(layer['config']['strides'][1])
        elif (layer['class_name']==MAX_POOL_1D_LAYER or layer['class_name']==AVG_POOL_1D_LAYER):
            height_array.append(layer['config']['pool_size'][0])
            width_array.append(1)
        else:
            height_array.append(0)

    return convert_array_to_string(height_array), convert_array_to_string(width_array)

def get_padding_string(input, padding_types):
    array=[]
    for layer in input['config']['layers']:
        if (layer['class_name']==MAX_POOL_2D_LAYER or layer['class_name']==AVG_POOL_2D_LAYER):
            array.append(padding_types[layer['config']['padding']])
        else:
            array.append(0)

    return convert_array_to_string(array)

def get_activation_function_string(input, activation_functions):
    """Returns a string containing an array of indices representing the activation function for each layer"""
    array = []
    for layer in input['config']['layers']:
        #? Dictionary activation_functions contains the mapping to the indices
        if (layer['class_name']==DENSE_LAYER):
            array.append(str(activation_functions[layer['config']['activation'].lower()]))
        else:
            array.append('0')
    return convert_array_to_string(array)

def get_bias_information(input):
    """Returns a string containing an array of bools indicating the usage of biases,
       a string containing an array of indices indicating the start position of biases for each layer,
       and a flattened array of weights values"""
    last_layer_values = 0
    count=0
    output= []
    bias_indices_array=[]
    use_bias_array = []
    for layer in input['config']['layers']:
        if (layer['class_name']==DENSE_LAYER):
            use_bias_array.append(int(layer['config']['use_bias']))
            bias_indices_array.append(int(last_layer_values))
            last_layer_values = last_layer_values + layer['config']['units']
            output.append(layer['bias_values'])
            count = count + 1
        else:
            use_bias_array.append(0)
            bias_indices_array.append(0)

    #? Flattening the array before returning
    bias_array = list(chain.from_iterable(output))
    return convert_array_to_string(use_bias_array), convert_array_to_string(bias_indices_array), bias_array

def get_weight_information(input, layerOutputHeight):
    """Returns a string containing an array of indices indicating the start position of weights for each layer,
       and a flattened array of weights values"""
    previous_layer_values = 0
    units_previous_dense_layer = 0
    count=0
    is_first_dense_layer=True
    output= []
    weights_indices_array=[]
    for layer in input['config']['layers']:
        if (layer['class_name']==DENSE_LAYER):
            weights_indices_array.append(int(previous_layer_values))
            previous_layer_values = previous_layer_values + int(layer['config']['units']) * layerOutputHeight[count]

            units_previous_dense_layer = int(layer['config']['units'])
            output.append(list(chain.from_iterable(layer['kernel_values'])))
        else:
            weights_indices_array.append(0)
        count = count + 1

    #? Flattening the array before returning
    weights_array = list(chain.from_iterable(output))
    return convert_array_to_string(weights_indices_array), weights_array