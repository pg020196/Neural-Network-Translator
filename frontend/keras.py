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

        count=0
        #? Adding batch_input_shape, units, weight- and bias-values for each layer to the generated json object
        for layer in model_json['config']['layers']:
            if (layer['class_name']=='Dense' or layer['class_name']=='Conv2D' or layer['class_name']=='Conv1D'):
                weights = model.layers[count].get_weights()[0]
                biases = model.layers[count].get_weights()[1]
                layer['kernel_values'] = weights.tolist()
                layer['bias_values'] = biases.tolist()

            count+=1

        #? Deleting unnecessary information from the json object
        del model_json['keras_version']
        del model_json['backend']

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