from plugin_collection import ConversionPlugin

class Float2Integer(ConversionPlugin):

    def __init__(self):
        super().__init__('float2int', 'Conversion from Float to Integer values')

    def process(self, input):
        #! TODO convert values from float to integer
        return input