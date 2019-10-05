from plugin_collection import FrontendPlugin
import json

#? redirecting stderr to /dev/null to hide information- and warning-messages from keras framework
#? afterwards setting stderr back to initial value
import sys, os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Model
from keras.models import load_model
sys.stderr = stderr

class Keras(FrontendPlugin):

    def __init__(self):
        super().__init__('keras', 'Keras Frontend Plugin')

    def transform_to_intermediate_format(self, input):
        #? redirecting stderr to /dev/null to hide information- and warning-messages from keras framework
        #? afterwards setting stderr back to initial value
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        model = load_model(input)
        model_json = json.loads(model.to_json())
        sys.stderr = stderr

        count=0
        last_batch_input_shape=None
        last_units=0

        for layer in model_json['config']['layers']:
            weights = model.layers[count].get_weights()[0]
            biases = model.layers[count].get_weights()[1]
            layer['kernel_values'] = weights.tolist()
            layer['bias_values'] = biases.tolist()

            if (last_batch_input_shape!=None):
                new_batch_input_shape = last_batch_input_shape
                new_batch_input_shape[1]= last_units
                layer['config']['batch_input_shape'] = new_batch_input_shape
            count+=1
            last_units = layer['config']['units']
            last_batch_input_shape = layer['config']['batch_input_shape']

        model_json.pop('keras_version', None)
        model_json.pop('backend', None)

        return model_json