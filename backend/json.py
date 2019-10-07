from plugin_collection import BackendPlugin
import json

class Json(BackendPlugin):
    """Json backend plugin returns the intermediate json format as string"""

    def __init__(self):
        super().__init__('json', 'JSON Backend Plugin', ['float2int'])

    def translate_to_native_code(self, input):
        """Returns the given json input as string representation"""
        return json.dumps(input)