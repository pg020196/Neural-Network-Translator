from plugin_collection import Plugin
import json

class Json(Plugin):

    def __init__(self):
        super().__init__()
        self.description = 'JSON Backend Plugin'
        self.identifier = 'json'

    def translate_to_native_code(self, input):
        return json.dumps(input)