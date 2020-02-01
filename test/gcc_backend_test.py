import unittest
import sys
import json
from backend.gcc.gcc import GCC

class TestGCCBackend(unittest.TestCase):
    """Test class for GCC Backend"""

    #? Naming scheme for test classes: [test_MethodName_StateUnderTest_ExpectedBehavior]

    intermediate = None

    def __init__(self, testname):
        super(TestGCCBackend, self).__init__(testname)

    def setUp(self):
        """Preparation for test cases"""
        self.intermediate = json.load(open('test/test_dense_2layer_input.json'))
        return super().setUp()

    def tearDown(self):
        """Clean up after test cases"""
        return super().tearDown()

    def test_buildMarkers_validIntermediateFormat_validMarkerDict(self):
        """Test case for build_markers function"""
        markers = GCC().build_markers(self.intermediate)
        #? Check if all markers were correctly added to the dictionary
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
#? Command: python -m test.gcc_backend_test

if __name__ == '__main__':
    #? Searching for all test cases in TestGCCBackend
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestGCCBackend)

    #? Adding all found test cases
    suite = unittest.TestSuite()
    for test_name in test_names:
        suite.addTest(TestGCCBackend(test_name))

    #? Running the test suite
    result = unittest.TextTestRunner().run(suite)
    sys.exit(not result.wasSuccessful())