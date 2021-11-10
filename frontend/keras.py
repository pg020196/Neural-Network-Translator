from plugin_collection import FrontendPlugin
import json

class Keras(FrontendPlugin):
    """Keras frontend plugin transforms given Keras h5-file to the intermediate format"""

    def __init__(self):
        super().__init__('keras', 'Keras Frontend Plugin')

    def transform_to_intermediate_format(self, input):
        """Returns the intermediate format represenation of the given h5-file"""
        from tensorflow import keras

        #? Loading the given model and transforming it to a json object
        model = keras.models.load_model(input)
        model_json = json.loads(model.to_json())

        #? Remove InputLayer from json file
        #? Currently, it does not provide any necessary attributes
        if(model_json['config']['layers'][0]['class_name']=='InputLayer'):
            model_json['config']['layers'].pop(0)

        count=0
        #? Adding batch_input_shape, units, weight- and bias-values for each layer to the generated json object
        for layer in model_json['config']['layers']:
            if (layer['class_name']=='Dense'):
                weights = model.layers[count].get_weights()[0]
                biases = model.layers[count].get_weights()[1]
                layer['kernel_values'] = weights.tolist()
                layer['bias_values'] = biases.tolist()

            #? Calculate padding size if necessary
            if(layer['class_name'] in ['AveragePooling1D','AveragePooling2D', 'MaxPooling1D', 'MaxPooling2D']):

                # In keras, padding has the value 'same' if it should be applied
                if (layer['config']['padding'] == 'same'):
                    #? Init padding size placeholder variables
                    vertical_padding = 0
                    horizontal_padding = 0

                    #? Formula for padding: padding = (pool_size - 1) / 2
                    #? Calculate vertical padding
                    vertical_padding = int((layer['config']['pool_size'][0] - 1) / 2)

                    #? Horizontal padding only exists in two or more dimensions
                    if (layer['class_name'] in ['AveragePooling2D', 'MaxPooling2D']):
                        #? Calculate horizontal padding  
                        horizontal_padding = int((layer['config']['pool_size'][1] - 1) / 2)

                    #? Insert padding values into intermediate format [top, bottom, left, right]
                    layer['config']['padding'] = [vertical_padding,horizontal_padding,vertical_padding,horizontal_padding]
                
                # In keras, padding has the value 'valid' if it should not be applied
                elif (layer['config']['padding'] == 'valid'):
                    # No padding should be applied
                    layer['config']['padding'] = [0,0,0,0]

            count+=1

            

        #? Deleting unnecessary information from the json object
        del model_json['keras_version']
        del model_json['backend']

        #? Removing unnecessary information from config object
        for layer in model_json['config']['layers']:
            layer['config'].pop('trainable', None)
            layer['config'].pop('kernel_initializer', None)
            layer['config'].pop('bias_initializer', None)
            layer['config'].pop('kernel_regularizer', None)
            layer['config'].pop('bias_regularizer', None)
            layer['config'].pop('activity_regularizer', None)
            layer['config'].pop('kernel_constraint', None)
            layer['config'].pop('bias_constraint', None)

        return model_json