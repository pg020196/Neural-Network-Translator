import unittest
import sys
import argparse
from test.backend_utils_test import TestBackendUtils
from test.arduino_backend_test import TestArduinoBackend

#? ############### INFO ###############
#? This script has to be run with -m switch from parent directory in order to get the imports right
#? Directory: Neural-Network-Translator (repository root directory)
#? Command: python -m test.complete_test_suite

parser = argparse.ArgumentParser(description='Neural Network Translator - TestRunner')
parser.add_argument('-c', '--com', type=str, required=True, help='COM of Arduino')
parser.add_argument('-b', '--baud', type=str, required=True, help='Baud rate of Arduino')
parser.add_argument('-m', '--model', type=str, required=True, help='h5-model')
args = parser.parse_args()

test_loader = unittest.TestLoader()

#? Running tests for backend_utils
print('######################### Running tests for backend_utils #########################')
backend_utils_test_names = test_loader.getTestCaseNames(TestBackendUtils)
suite = unittest.TestSuite()
for test_name in backend_utils_test_names:
    suite.addTest(TestBackendUtils(test_name))

result_backend_utils = unittest.TextTestRunner().run(suite)
print()

#? Running tests for Arduino backend
print('######################### Running tests for Arduino backend #########################')
arduino_backend_test_names = test_loader.getTestCaseNames(TestArduinoBackend)
suite = unittest.TestSuite()
for test_name in arduino_backend_test_names:
    suite.addTest(TestArduinoBackend(test_name, args.com, args.baud, args.model))

result_arduino_backend = unittest.TextTestRunner().run(suite)

sys.exit(not (result_backend_utils.wasSuccessful() and result_arduino_backend.wasSuccessful()))