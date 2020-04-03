import traceback
import argparse
import json
from plugin_collection import PluginCollection
from jsonschema import validate,ValidationError

def get_available_plugins(plugins):
    """ Returns a string containing a list of available plugins in the given PluginCollection"""
    av_plugins=''
    for plugin in plugins:
        av_plugins += plugin.identifier + ', '
    return av_plugins[:-2]

#? Collecting the available plugins in the corresponding folders
frontend_plugins = PluginCollection('frontend')
conversion_plugins = PluginCollection('conversion')
backend_plugins = PluginCollection('backend')

#? Parsing command line arguments
parser = argparse.ArgumentParser(description='Translates high-level neural network model to native code for specified backend')
parser.add_argument('-f', '--frontend', type=str, required=True, help='Frontend type of the input file, available at the moment: '+ get_available_plugins(frontend_plugins.plugins))
parser.add_argument('-b', '--backend', type=str, required=True, help='Backend type to translate into, available at the moment: '+ get_available_plugins(backend_plugins.plugins))
parser.add_argument('-c', '--conversions', nargs='+', help='Conversions to be performed on data, available at the moment: '+ get_available_plugins(conversion_plugins.plugins))
parser.add_argument('-i', '--input', type=str, required=True, help='Input file containing the neural network model')
parser.add_argument('-o', '--output', type=str, required=True, help='Output file to write to')
parser.add_argument('-e', '--executable', type=str, help='Path to an executable file which contains the prediction call, when set the given file will be copied into the output directory')
args = parser.parse_args()

try:
    #? Searching for the correct plugins to process the given input
    frontend = frontend_plugins.get_plugin(args.frontend.lower())
    backend = backend_plugins.get_plugin(args.backend.lower())

    print('Converting inputfile "' + args.input + '" to intermediate format with plugin "' + frontend.identifier + '"')
    #? Transforming the input file to the intermediate format
    intermediate = frontend.transform_to_intermediate_format(args.input)

    #? Validating the produced intermediate format with the json schema in order to check the correctness of the result
    with open('intermediate.schema.json') as json_file:
        schema = json.load(json_file)
        validate(intermediate, schema)

    #? If conversion plugins were defined within the command line arguments those are executed
    if (args.conversions is not None):
        for conversion in args.conversions:
            conv_plugin = conversion_plugins.get_plugin(conversion)
            print('Performing conversion: "' + conv_plugin.description + '"')
            intermediate = conv_plugin.process(intermediate)

    #? If the selected backend plugin has any prerequisites those conversions are executed too
    if (backend.prerequisites is not None):
        for prerequisite in backend.prerequisites:
            conv_plugin = conversion_plugins.get_plugin(prerequisite)
            print('Performing required conversion by backend plugin "' + backend.identifier + '": "' + conv_plugin.description + '"')
            intermediate = conv_plugin.process(intermediate)

    print('Translating intermediate format to native code with plugin "' + backend.identifier + '"')
    exec_file = None
    if (args.executable is not None):
        exec_file = args.executable
    #? Translating the produced intermediate format to native code of the backend plugin
    backend.translate_to_native_code(intermediate, args.output, exec_file)

    print('Translation of input-file "' + args.input + '" to output-file "' + args.output + '" successfully completed')

#? Error Handling
except ValidationError:
    print('Output of frontend plugin "' + frontend.identifier + '" does not match JSON schema.')
except NotImplementedError:
    print('Selected frontend/backend is not available')
except IOError as ioerr:
    print('Error occurred while opening the file: ')
    print(traceback.format_exception_only(type(ioerr), ioerr))
except Exception as err:
    print('An error occurred: ')
    print(traceback.format_exception_only(type(err), err))
