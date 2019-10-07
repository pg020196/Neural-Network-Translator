from itertools import chain

def get_number_of_layers(input):
    """Returns the number of layers in the neural network"""
    count=0
    for layer in input['config']['layers']:
        count = count + 1
    return count + 1

def get_bias_array(input, numberLayers):
    """Returns a flattened array of bias values"""
    count=0
    output= [None] * numberLayers
    for layer in input['config']['layers']:
        output[count] = layer['bias_values']
        count = count + 1
    #? Flattening the array before returning
    return list(chain.from_iterable(output))

def get_weights_array(input, numberLayers):
    """Returns a flattened array of weights values"""
    count=0
    output= [None] * numberLayers
    for layer in input['config']['layers']:
        output[count] = list(chain.from_iterable(layer['kernel_values']))
        count = count + 1
    #? Flattening the array before returning
    return list(chain.from_iterable(output))

def replace_markers(file, markers):
    """ Replaces all given markes in given file with their respective value"""
    for marker, value in markers.items():
        file = file.replace(marker,str(value))
    return file

def read_marker_file(filename):
    """ Reads given filename"""
    with open(filename, 'r') as file:
        return file.read()

def build_activation_function_string(input, activation_functions):
        """Returns a string containing an array of indices representing the activation function for each layer"""
        string = '{'
        for layer in input['config']['layers']:
            #? Dictionary activation_functions contains the mapping to the indices
            string = string + str(activation_functions[layer['config']['activation'].lower()]) + ','
        return string[:-1] + '}'

def build_use_bias_string(input):
    """Returns a string containing an array of bools indicating the usage of biases"""
    string = '{'
    for layer in input['config']['layers']:
        string = string + str(layer['config']['use_bias']).lower() + ','
    return string[:-1] + '}'

def build_units_in_layers_string(input):
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

def build_layer_types_string(input, layer_types):
    """Returns a string containing an array of indices representing the layer type for each layer"""
    string = '{'
    for layer in input['config']['layers']:
        #? Dictionary layer_types contains the mapping to the indices
        string = string + str(layer_types[layer['class_name'].lower()]) + ','
    return string[:-1] + '}'

def build_indices_weights_string(input):
    """Returns a string containing an array of indices indicating the start position of weights for each layer"""
    last_layer_values = 0
    string = '{'
    for layer in input['config']['layers']:
        string = string + str(last_layer_values) + ','
        last_layer_values = last_layer_values + int(layer['config']['units']) * int(layer['config']['batch_input_shape'][1])
    return string[:-1] + '}'

def build_indices_bias_string(input):
    """Returns a string containing an array of indices indicating the start position of biases for each layer"""
    last_layer_values = 0
    string = '{'
    for layer in input['config']['layers']:
        string = string + str(last_layer_values) + ','
        last_layer_values = last_layer_values + layer['config']['units']
    return string[:-1] + '}'

def convert_array_to_string(array):
    """Returns a string containing the given array"""
    string = '{'
    for value in array:
        string = string + str(value) + ','
    return string[:-1] + '}'