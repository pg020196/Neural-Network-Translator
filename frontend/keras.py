from plugin_collection import FrontendPlugin
import json

class Keras(FrontendPlugin):
    """Keras frontend plugin transforms given Keras h5-file to the intermediate format"""

    def __init__(self):
        super().__init__('keras', 'Keras Frontend Plugin')

    def transform_to_intermediate_format(self, input):
        """Returns the intermediate format represenation of the given h5-file"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.models import load_model

        #? Loading the given model and transforming it to a json object
        model = load_model(input)
        model_json = json.loads(model.to_json())

        count=0
        last_batch_input_shape=None
        last_units=0

        #? Adding batch_input_shape, units, weight- and bias-values for each layer to the generated json object
        for layer in model_json['config']['layers']:
            weights = model.layers[count].get_weights()[0]
            biases = model.layers[count].get_weights()[1]
            layer['kernel_values'] = weights.tolist()
            layer['bias_values'] = biases.tolist()

            if (last_batch_input_shape!=None):
                new_batch_input_shape = last_batch_input_shape
                new_batch_input_shape[1]= last_units
                layer['config']['batch_input_shape'] = new_batch_input_shape

            last_units = layer['config']['units']
            last_batch_input_shape = layer['config']['batch_input_shape']

            count+=1

        #? Deleting unnecessary information from the json object
        del model_json['keras_version']
        del model_json['backend']

        for layer in model_json['config']['layers']:
            del layer['config']['trainable']
            del layer['config']['kernel_initializer']
            del layer['config']['bias_initializer']
            del layer['config']['kernel_regularizer']
            del layer['config']['bias_regularizer']
            del layer['config']['activity_regularizer']
            del layer['config']['kernel_constraint']
            del layer['config']['bias_constraint']

        return model_json