from plugin_collection import PluginCollection
import argparse
from jsonschema import validate,ValidationError
import json

def get_available_plugins(plugins):
    av_plugins=''
    for plugin in plugins:
        av_plugins += plugin.identifier + ", "
    return av_plugins[:-2]

frontend_plugins = PluginCollection('frontend')
backend_plugins = PluginCollection('backend')

parser = argparse.ArgumentParser(description='Translates high-level neural network model to native code for specified backend')
parser.add_argument('-f', '--frontend', type=str, required=True, help='Frontend type of the input file, available at the momement: '+ get_available_plugins(frontend_plugins.plugins))
parser.add_argument('-b', '--backend', type=str, required=True, help='Backend type to translate into, available at the momement: '+ get_available_plugins(backend_plugins.plugins))
parser.add_argument('-i', '--input', type=str, required=True, help='Input file containing the neural network model')
parser.add_argument('-o', '--output', type=str, required=True, help='Output file to write to')
args = parser.parse_args()

try:
    frontend = frontend_plugins.get_plugin(args.frontend.lower())
    backend = backend_plugins.get_plugin(args.backend.lower())
except NotImplementedError:
    print('Selected frontend/backend is not available')
    exit()

try:
    intermediate = frontend.transform_to_intermediate_format(args.input)

    with open('intermediate.schema.json') as json_file:
        schema = json.load(json_file)
        validate(intermediate, schema)

    native_code = backend.translate_to_native_code(intermediate)
except ValidationError:
    print('Output of frontend plugin "' + frontend.identifier + '" does not match JSON schema.')
    exit()

with open(args.output, 'w') as file:
    file.write(native_code)


