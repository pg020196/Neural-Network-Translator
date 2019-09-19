from plugin_collection import BackendPlugin
import json

class Json(BackendPlugin):

    def __init__(self):
        super().__init__('json', 'JSON Backend Plugin', ['float2int'])

    def translate_to_native_code(self, input):
        return json.dumps(input)