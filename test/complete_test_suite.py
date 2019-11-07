import unittest
import sys
import argparse
from test.backend_utils_test import TestBackendUtils
from test.gcc_backend_test import TestGCCBackend

#? ############### INFO ###############
#? This script has to be run with -m switch from parent directory in order to get the imports right
#? Directory: Neural-Network-Translator (repository root directory)
#? Command: python -m test.complete_test_suite

test_loader = unittest.TestLoader()

#? Running tests for backend_utils
print('######################### Running tests for backend_utils #########################')

#? Finding all test cases in TestBackendUtils and executing the test suite
backend_utils_test_names = test_loader.getTestCaseNames(TestBackendUtils)
suite = unittest.TestSuite()
for test_name in backend_utils_test_names:
    suite.addTest(TestBackendUtils(test_name))

result_backend_utils = unittest.TextTestRunner().run(suite)
print()

#? Running tests for GCC backend
print('######################### Running tests for GCC backend #########################')

#? Finding all test cases in TestGCCBackend and executing the test suite
gcc_backend_test_names = test_loader.getTestCaseNames(TestGCCBackend)
suite = unittest.TestSuite()
for test_name in gcc_backend_test_names:
    suite.addTest(TestGCCBackend(test_name))

result_gcc_backend = unittest.TextTestRunner().run(suite)


sys.exit(not (result_backend_utils.wasSuccessful() and result_gcc_backend.wasSuccessful()))