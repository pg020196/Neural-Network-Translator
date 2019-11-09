from plugin_collection import BackendPlugin
import json
import os

class Json(BackendPlugin):
    """Json backend plugin returns the intermediate json format as string"""

    def __init__(self):
        super().__init__('json', 'JSON Backend Plugin', ['float2int'])

    def translate_to_native_code(self, input, outputfile, exec_file):
        """Returns the given json input as string representation"""
        out_name, out_ext = os.path.splitext(outputfile)
        out_dir = '_out/' + out_name

        #? Creating directory if not existing
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out = out_dir + '/' + outputfile
        if (out_ext!='.json'):
            out = out + '.json'

        with open(out, 'w') as file:
            file.write(json.dumps(input))