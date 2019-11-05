import unittest
import serial
import numpy as np
import argparse
import sys
import json
import tensorflow as tf
from backend.arduino import Arduino

class TestArduinoBackend(unittest.TestCase):

    inputs = []
    arduino = None
    precision = 8
    modelPath = '../diabetes_model.h5'
    intermediate = None
    com = 'COM3'
    baud = 9600

    def __init__(self, testname, com, baud, model):
        super(TestArduinoBackend, self).__init__(testname)
        self.modelPath = model
        self.intermediate = json.load(open('test/test_dense_2layer_input.json'))
        self.com = com
        self.baud = baud

    def setUp(self):
        self.arduino = serial.Serial(self.com, self.baud, timeout=5)
        self.inputs.append('6,148,72,35,0,33.6,0.627,50') #1
        self.inputs.append('10,168,74,0,0,38,0.537,34') #1
        self.inputs.append('0,101,65,28,0,24.6,0.237,22') #0
        self.inputs.append('4,97,60,23,0,28.2,0.443,22') #0
        self.inputs.append('7,125,86,0,0,37.6,0.304,51') #0
        self.inputs.append('11,120,80,37,150,42.3,0.785,48') #1
        return super().setUp()

    def tearDown(self):
        self.arduino.close()
        return super().tearDown()

    def test_arduino_translate_to_native_code(self):
        results_arduino = []

        for input_values in self.inputs:
            self.arduino.readline()
            self.arduino.write(input_values.encode())
            self.arduino.readline()
            result = self.arduino.readline()
            self.arduino.readline()
            self.arduino.readline()
            results_arduino.append(result.decode().rstrip('\r\n'))

        results_framework = []
        model = tf.keras.models.load_model(self.modelPath)

        for input_values in self.inputs:
            array = str(input_values).split(',')
            array = np.array(array).reshape(1,len(array))
            results_framework.append(model.predict(array.astype(np.float)))

        for i in range(0, len(results_arduino)):
            counter = 0
            dot_index = 0
            for digit in str(float(results_framework[i][0][0])):
                if (digit=='.'):
                    dot_index=counter
                if (digit==str(float(results_arduino[i]))[counter]):
                    counter=counter+1
                else:
                    break

            if ((counter-dot_index +1)<self.precision):
                self.assertTrue(False)

        self.assertTrue(True)

    def test_build_markers(self):
        markers = Arduino().build_markers(self.intermediate)
        self.assertTrue('###numberLayers###' in markers and
                        '###dimNumberLayers###' in markers and
                        '###layerTypes###' in markers and
                        '###layerTypes###' in markers and
                        '###layerOutputWidth###' in markers and
                        '###layerOutputHeight###' in markers and
                        '###activationFunctions###' in markers and
                        '###weights###' in markers and
                        '###dimWeights###' in markers and
                        '###indicesWeights###' in markers and
                        '###bias###' in markers and
                        '###dimBias###' in markers and
                        '###indicesBias###' in markers and
                        '###useBias###' in markers and
                        '###poolWidth###' in markers and
                        '###poolHeight###' in markers and
                        '###horizontalStride###' in markers and
                        '###verticalStride###' in markers and
                        '###padding###' in markers)

#? ############### INFO ###############
#? This script has to be run with -m switch from parent directory in order to get the imports right
#? Directory: Neural-Network-Translator (repository root directory)
#? Command: python -m test.arduino_backend_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arduino unit test')
    parser.add_argument('-c', '--com', type=str, required=True, help='COM of Arduino')
    parser.add_argument('-b', '--baud', type=str, required=True, help='Baud rate of Arduino')
    parser.add_argument('-m', '--model', type=str, required=True, help='h5-model')
    args = parser.parse_args()

    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestArduinoBackend)

    suite = unittest.TestSuite()
    for test_name in test_names:
        suite.addTest(TestArduinoBackend(test_name, args.com, args.baud, args.model))

    result = unittest.TextTestRunner().run(suite)
    sys.exit(not result.wasSuccessful())