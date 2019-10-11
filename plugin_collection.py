import inspect
import os
import pkgutil

class Plugin(object):
    """Base plugin that every plugin must inherit from"""

    def __init__(self, identifier, description):
        self.description = description
        self.identifier = identifier

class FrontendPlugin(Plugin):
    """Base class that each frontend plugin must inherit from"""

    def __init__(self, identifier, description):
        super().__init__(identifier, description)

    def transform_to_intermediate_format(self, input):
        """Transforms given input into intermediate format and returns output as string"""
        raise NotImplementedError

class BackendPlugin(Plugin):
    """Base class that each backend plugin must inherit from"""

    def __init__(self, identifier, description, prerequisites):
        super().__init__(identifier, description)
        self.prerequisites = prerequisites

    def translate_to_native_code(self, input):
        """Translates given input in intermediate format to native code"""
        raise NotImplementedError

class ConversionPlugin(Plugin):
    """Base class that each conversion plugin must inherit from"""
    def __init__(self, identifier, description):
        super().__init__(identifier, description)

    def process(self, input):
        """Processes the given input and performs a specific conversion, returns processed output"""
        raise NotImplementedError

class PluginCollection(object):
    """Manages a list of plugins available in a given directory"""

    def __init__(self, plugin_package):
        self.plugin_package = plugin_package
        self.reload_plugins()

    def reload_plugins(self):
        """Loads all available plugins"""
        self.plugins = []
        self.seen_paths = []
        self.search_for_plugins(self.plugin_package)


    def get_plugin(self, plugin_identifier):
        """Searches for a plugin with a given identifier in the managed plugin list"""
        for plugin in self.plugins:
            if (plugin.identifier == plugin_identifier):
                return plugin
        raise NotImplementedError

    def search_for_plugins(self, package):
        """Searches the given directory for available plugins"""

        imported_package = __import__(package, fromlist=[''])

        #? Recursivly searching for classes inheriting from the specified
        for _, pluginname, ispkg in pkgutil.iter_modules(imported_package.__path__, imported_package.__name__ + '.'):
            if not ispkg:
                plugin_module = __import__(pluginname, fromlist=[''])
                clsmembers = inspect.getmembers(plugin_module, inspect.isclass)
                for (_, c) in clsmembers:
                    #? Only add classes that are a sub class of Frontend-, Backend- or ConversionPlugin, but not the class itself
                    if issubclass(c, Plugin) & (c is not Plugin and c is not FrontendPlugin and c is not BackendPlugin and c is not ConversionPlugin):
                        self.plugins.append(c())
