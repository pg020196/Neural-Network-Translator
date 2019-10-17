import unittest
import serial
import numpy as np
import argparse
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

class TestArduinoBackend(unittest.TestCase):

    inputs = []
    arduino = None
    precision = 8

    def __init__(self, testname, com, baud):
        super(TestArduinoBackend, self).__init__(testname)
        self.arduino = serial.Serial(com,baud, timeout=5)

    def setUp(self):
        self.inputs.append('6,148,72,35,0,33.6,0.627,50') #1
        self.inputs.append('10,168,74,0,0,38,0.537,34') #1
        self.inputs.append('0,101,65,28,0,24.6,0.237,22') #0
        self.inputs.append('4,97,60,23,0,28.2,0.443,22') #0
        self.inputs.append('7,125,86,0,0,37.6,0.304,51') #0
        self.inputs.append('11,120,80,37,150,42.3,0.785,48') #1
        return super().setUp()

    def tearDown(self):
        return super().tearDown()

    def test_results(self):
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
        model = load_model('../diabetes_model.h5')

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arduino unit test')
    parser.add_argument('-c', '--com', type=str, required=True, help='COM of Arduino')
    parser.add_argument('-b', '--baud', type=str, required=True, help='Baud rate of Arduino')
    args = parser.parse_args()

    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestArduinoBackend)

    suite = unittest.TestSuite()
    for test_name in test_names:
        suite.addTest(TestArduinoBackend(test_name, args.com, args.baud))

    result = unittest.TextTestRunner().run(suite)
    sys.exit(not result.wasSuccessful())