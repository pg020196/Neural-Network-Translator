import inspect
import os
import pkgutil

class Plugin(object):
    # Base class that each plugin must inherit from

    def __init__(self):
        self.description = 'UNKNOWN'
        self.identifier = 'IDENTIFIER'

    def transform_to_intermediate_format(self, input):
        raise NotImplementedError

    def translate_to_native_code(self, input):
        raise NotImplementedError

class PluginCollection(object):
    # This class will read the plugins package for modules that contain a class definition that is inheriting from the Plugin class

    def __init__(self, plugin_package):
        self.plugin_package = plugin_package
        self.reload_plugins()

    def reload_plugins(self):
        self.plugins = []
        self.seen_paths = []
        self.search_for_plugins(self.plugin_package)


    def get_plugin(self, plugin_identifier):
        for plugin in self.plugins:
            if (plugin.identifier == plugin_identifier):
                return plugin
        raise NotImplementedError

    def search_for_plugins(self, package):
        # Searching the supplied package for plugins

        imported_package = __import__(package, fromlist=[''])

        for _, pluginname, ispkg in pkgutil.iter_modules(imported_package.__path__, imported_package.__name__ + '.'):
            if not ispkg:
                plugin_module = __import__(pluginname, fromlist=[''])
                clsmembers = inspect.getmembers(plugin_module, inspect.isclass)
                for (_, c) in clsmembers:
                    # Only add classes that are a sub class of Plugin, but not Plugin itself
                    if issubclass(c, Plugin) & (c is not Plugin):
                        self.plugins.append(c())
