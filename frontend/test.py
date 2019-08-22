from plugin_collection import Plugin

class Test(Plugin):

    def __init__(self):
        super().__init__()
        self.description = 'test Plugin'
        self.identifier = 'test'

    def transform_to_intermediate_format(self, input):
        return input