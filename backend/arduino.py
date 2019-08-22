from plugin_collection import Plugin
import json

class Keras(Plugin):

    def __init__(self):
        super().__init__()
        self.description = 'Arduino Backend Plugin'
        self.identifier = 'arduino'

    def translate_to_native_code(self, input):
        #! TODO translate input to native code
        return json.dumps(input)