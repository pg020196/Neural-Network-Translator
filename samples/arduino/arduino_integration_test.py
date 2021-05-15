import unittest
import serial
import numpy as np
import argparse
import sys
import tensorflow as tf

class ArduinoIntegrationTest(unittest.TestCase):
    """Test class for Arduino Backend"""

    inputs = []
    arduino = None
    precision = 6
    modelPath = '../diabetes_model.h5'
    com = 'COM3'
    baud = 9600

    def __init__(self, testname, com, baud, model):
        super(ArduinoIntegrationTest, self).__init__(testname)
        self.modelPath = model
        self.com = com
        self.baud = baud

    def setUp(self):
        """Preparation for test cases"""
        self.arduino = serial.Serial(self.com, self.baud, timeout=5)
        self.inputs.append('6,148,72,35,0,33.6,0.627,50') #1
        self.inputs.append('10,168,74,0,0,38,0.537,34') #1
        self.inputs.append('0,101,65,28,0,24.6,0.237,22') #0
        self.inputs.append('4,97,60,23,0,28.2,0.443,22') #0
        self.inputs.append('7,125,86,0,0,37.6,0.304,51') #0
        self.inputs.append('11,120,80,37,150,42.3,0.785,48') #1
        return super().setUp()

    def tearDown(self):
        """Clean up after test cases"""
        self.arduino.close()
        return super().tearDown()

    def test_arduino_translate_to_native_code(self):
        """ Test case for translate to native code function"""
        results_arduino = []

        #? Sending test inputs to Arduino and collecting results
        for input_values in self.inputs:
            self.arduino.readline()
            self.arduino.write(input_values.encode())
            self.arduino.readline()
            result = self.arduino.readline()
            self.arduino.readline()
            self.arduino.readline()
            prediction = float(result.decode().rstrip('\r\n').strip())
            results_arduino.append(prediction)

        results_framework = []
        model = tf.keras.models.load_model(self.modelPath)

        #? Collecting results from framework prediction for test inputs
        for input_values in self.inputs:
            array = str(input_values).split(',')
            array = np.array(array).reshape(1,len(array))
            prediction = model.predict(array.astype(np.float))[0][0]
            results_framework.append(prediction)

        #? Checking if results from framework and arduino differ or not
        for i in range(0, len(results_arduino)):
            self.assertAlmostEqual(results_arduino[i], results_framework[i], self.precision)

        self.assertTrue(True)

#? ############### INFO ###############
#? This script has to be run with -m switch from parent directory in order to get the imports right
#? Directory: Neural-Network-Translator (repository root directory)
#? Command: python -m samples.arduino.arduino_integration_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arduino integration test')
    parser.add_argument('-c', '--com', type=str, required=True, help='COM of Arduino')
    parser.add_argument('-b', '--baud', type=str, required=True, help='Baud rate of Arduino')
    parser.add_argument('-m', '--model', type=str, required=True, help='h5-model')
    args = parser.parse_args()

    #? Searching for all test cases in ArduinoIntegrationTest
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(ArduinoIntegrationTest)

    #? Adding all found test cases
    suite = unittest.TestSuite()
    for test_name in test_names:
        suite.addTest(ArduinoIntegrationTest(test_name, args.com, args.baud, args.model))

    #? Running the test suite
    result = unittest.TextTestRunner().run(suite)
    sys.exit(not result.wasSuccessful())