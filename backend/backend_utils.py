from itertools import chain

DENSE_LAYER = 'Dense'
CONV_2D_LAYER = 'Conv2D'
CONV_1D_LAYER = 'Conv1D'
MAX_POOL_2D_LAYER = 'MaxPooling2D'
AVG_POOL_2D_LAYER = 'AvgPooling2D'
FLATTEN_LAYER = 'Flatten'

def replace_markers(file, markers):
    """ Replaces all given markes in given file with their respective value"""
    for marker, value in markers.items():
        file = file.replace(marker,str(value))
    return file

def read_marker_file(filename):
    """ Reads given filename"""
    with open(filename, 'r') as file:
        return file.read()

def convert_array_to_string(array):
    """Returns a string containing the given array"""
    string = '{'
    for value in array:
        string = string + str(value) + ','
    return string[:-1] + '}'

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
    height_array = []
    last_output_height=0
    width_array = []
    last_output_width=0

    for layer in input['config']['layers']:
        input_height = last_output_height
        input_width = last_output_width

        if 'batch_input_shape' in layer['config']:
            height_array.append(layer['config']['batch_input_shape'][1])
            input_height = layer['config']['batch_input_shape'][1]

            last_output_height = layer['config']['batch_input_shape'][1]

            if (len(layer['config']['batch_input_shape'])>2):
                width_array.append(layer['config']['batch_input_shape'][2])
                input_width = layer['config']['batch_input_shape'][2]

                last_output_width = layer['config']['batch_input_shape'][2]
            else:
                width_array.append(1)
                input_width = 1

                last_output_width = 1

        if (layer['class_name']==DENSE_LAYER):
            act_height = len(layer['kernel_values'][0])
            act_width = 1
        if (layer['class_name']==FLATTEN_LAYER):
            act_height = last_output_height * last_output_width
            act_width = 1
        if (layer['class_name']==AVG_POOL_2D_LAYER or layer['class_name']==MAX_POOL_2D_LAYER):
            vertical_padding = (layer['config']['pool_size'][0] - 1) / 2
            act_height = ((input_height - layer['config']['pool_size'][0] + 2 * vertical_padding) / layer['config']['strides'][0]) + 1
            horizontal_padding = (layer['config']['pool_size'][1] - 1) / 2
            act_width = ((input_width - layer['config']['pool_size'][1] + 2 * horizontal_padding) / layer['config']['strides'][1]) + 1

        height_array.append(act_height)
        width_array.append(act_width)

        last_output_height = act_height
        last_output_width = act_width

    return height_array, width_array

def get_pool_size_strings(input):
    width_array=[]
    height_array=[]
    for layer in input['config']['layers']:
        if (layer['class_name']==MAX_POOL_2D_LAYER or layer['class_name']==AVG_POOL_2D_LAYER):
            width_array.append(layer['config']['pool_size'][1])
            height_array.append(layer['config']['pool_size'][0])
        else:
            width_array.append(0)
            height_array.append(0)
    return convert_array_to_string(width_array), convert_array_to_string(height_array)

def get_strides_strings(input):
    width_array=[]
    height_array=[]
    for layer in input['config']['layers']:
        if (layer['class_name']==MAX_POOL_2D_LAYER or layer['class_name']==AVG_POOL_2D_LAYER):
            width_array.append(layer['config']['strides'][1])
            height_array.append(layer['config']['strides'][0])
        else:
            width_array.append(0)
            height_array.append(0)
    return convert_array_to_string(width_array), convert_array_to_string(height_array)

def get_padding_string(input):
    array=[]
    for layer in input['config']['layers']:
        if (layer['class_name']==MAX_POOL_2D_LAYER or layer['class_name']==AVG_POOL_2D_LAYER):
            array.append(layer['config']['padding'])
        else:
            array.append('None')

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
            use_bias_array.append(str(int(layer['config']['use_bias'])))
            bias_indices_array.append(str(last_layer_values))
            last_layer_values = last_layer_values + layer['config']['units']
            output.append(layer['bias_values'])
            count = count + 1
        else:
            use_bias_array.append('0')
            bias_indices_array.append('0')

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
            weights_indices_array.append(str(previous_layer_values))

            # if ('batch_input_shape' in layer['config']):
            #     units_previous_dense_layer = layer['config']['batch_input_shape'][1]
            # elif (is_first_dense_layer):
            #     units_previous_dense_layer = layer['config']['units']
            #     is_first_dense_layer=False

            #last_layer_values = last_layer_values + int(layer['config']['units']) * int(layer['config']['batch_input_shape'][1])
            # TODO: ask Phiipp if this is correct

            previous_layer_values = previous_layer_values + int(layer['config']['units']) * layerOutputHeight[count]

            units_previous_dense_layer = int(layer['config']['units'])
            output.append(list(chain.from_iterable(layer['kernel_values'])))
        else:
            weights_indices_array.append('0')
        count = count + 1

    #? Flattening the array before returning
    weights_array = list(chain.from_iterable(output))
    return convert_array_to_string(weights_indices_array), weights_array

def get_units_in_layer_string(input):
    units_array = []
    for layer in input['config']['layers']:
        if (layer['class_name']==DENSE_LAYER):
            if 'batch_input_shape' in layer['config']:
                units_array.append(layer['config']['batch_input_shape'][1])

            units_array.append(layer['config']['units'])
        else:
            if 'batch_input_shape' in layer['config']:
                units_array.append('0')
            units_array.append('0')

    return convert_array_to_string(units_array)