from plugin_collection import BackendPlugin
import json

class Arduino(BackendPlugin):

    def __init__(self):
        super().__init__('arduino','Arduino Backend Plugin', None)

    def translate_to_native_code(self, input):
        #! TODO translate input to native code
        return json.dumps(input)