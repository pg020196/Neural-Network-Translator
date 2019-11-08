import unittest
import sys
import json
import backend.gcc.backend_utils as backend_utils
from backend.gcc.gcc import GCC

class TestBackendUtils(unittest.TestCase):

    dense_2layer_input = None
    dense_3layer_input = None
    mnist_flatten_input = None

    def __init__(self, testname):
        super(TestBackendUtils, self).__init__(testname)

    def setUp(self):
        """Preparation for test cases"""
        self.dense_2layer_input = json.load(open('test/test_dense_2layer_input.json'))
        self.dense_3layer_input = json.load(open('test/test_dense_3layer_input.json'))
        self.mnist_flatten_input = json.load(open('test/test_mnist_flatten_input.json'))
        return super().setUp()

    def tearDown(self):
        """Clean up after test cases"""
        return super().tearDown()

    def test_replace_markers(self):
        """Test case for replace_markers function"""
        string = 'abc ###marker### def'
        markers = {'###marker###':'hij'}
        self.assertTrue(backend_utils.replace_markers(string, markers) == 'abc hij def')

    def test_replace_markers_multiple(self):
        """Test case for replace_markers function with multiple replacments"""
        string = 'abc ###marker### def ###marker###'
        markers = {'###marker###':'hij'}
        self.assertTrue(backend_utils.replace_markers(string, markers) == 'abc hij def hij')

    def test_replace_markers_multiple_different(self):
        """Test case for replace_markers function with 2 markers"""
        string = 'abc ###marker### def ###marker2###'
        markers = {'###marker###':'hij', '###marker2###':'klm'}
        self.assertTrue(backend_utils.replace_markers(string, markers) == 'abc hij def klm')

    def test_replace_markers_multiple_different_direct_neighbouring(self):
        """Test case for replace_markers function with neighbouring markers"""
        string = 'abc def ###marker######marker2###'
        markers = {'###marker###':'hij', '###marker2###':'klm'}
        self.assertTrue(backend_utils.replace_markers(string, markers) == 'abc def hijklm')

    def test_replace_markers_not_existent(self):
        """Test case for replace_markers function not existent marker"""
        string = 'abc ###marker### def'
        markers = {'###nowhere###':'hij'}
        self.assertTrue(backend_utils.replace_markers(string, markers) == 'abc ###marker### def')

    def test_convert_array_to_string(self):
        """Test case for convert_array_to_string function"""
        self.assertTrue(backend_utils.convert_array_to_string([1,2,3,4]) == '{1,2,3,4}')

    def test_convert_array_to_string_float(self):
        """Test case for convert_array_to_string function with floats"""
        self.assertTrue(backend_utils.convert_array_to_string([1.0,2.0,3.0,4.0]) == '{1.0,2.0,3.0,4.0}')

    def test_convert_array_to_string_float_int_mixed(self):
        """Test case for convert_array_to_string function with ints and floats"""
        self.assertTrue(backend_utils.convert_array_to_string([1.0,2,3,4.0]) == '{1.0,2,3,4.0}')

    def test_convert_array_to_string_one_element(self):
        """Test case for convert_array_to_string function with only one element"""
        self.assertTrue(backend_utils.convert_array_to_string([1]) == '{1}')

    def test_get_number_of_layers_dense2(self):
        """Test case for get_number_of_layers function in 2 layer dense network"""
        self.assertTrue(backend_utils.get_number_of_layers(self.dense_2layer_input) == 3)

    def test_get_number_of_layers_dense3(self):
        """Test case for get_number_of_layers function in 3 layer dense network"""
        self.assertTrue(backend_utils.get_number_of_layers(self.dense_3layer_input) == 4)

    def test_get_layer_types_string_dense2(self):
        """Test case for get_layer_types_string function in 2 layer dense network"""
        self.assertTrue(backend_utils.get_layer_types_string(self.dense_2layer_input, GCC.layer_types) == '{1,1}')

    def test_get_layer_types_string_flatten_avgpool3d(self):
        """Test case for get_number_of_layers function with flattten and avgpooling3d layers"""
        self.dense_2layer_input['config']['layers'][0]['class_name'] = 'Flatten'
        self.dense_2layer_input['config']['layers'][1]['class_name'] = 'AvgPooling3D'
        self.assertTrue(backend_utils.get_layer_types_string(self.dense_2layer_input, GCC.layer_types) == '{2,4}')

    def test_get_output_dimensions_dense2(self):
        """Test case for get_output_dimensions function in 2 layer dense network"""
        heights, widths = backend_utils.get_output_dimensions(self.dense_2layer_input)
        self.assertTrue(backend_utils.convert_array_to_string(heights) == '{8,8,1}'
                    and backend_utils.convert_array_to_string(widths) == '{1,1,1}')

    def test_get_output_dimensions_dense3(self):
        """Test case for get_output_dimensions function in 3 layer dense network"""
        heights, widths = backend_utils.get_output_dimensions(self.dense_3layer_input)
        self.assertTrue(backend_utils.convert_array_to_string(heights) == '{8,16,8,1}'
                    and backend_utils.convert_array_to_string(widths) == '{1,1,1,1}')

    def test_get_output_dimensions_flatten_dense_dropout(self):
        """Test case for get_output_dimensions function with flatten, dense and dropout layers"""
        heights, widths = backend_utils.get_output_dimensions(self.mnist_flatten_input)
        self.assertTrue(backend_utils.convert_array_to_string(heights) == '{28,784,128,128,10}'
                    and backend_utils.convert_array_to_string(widths) =='{28,1,1,1,1}')

    def test_get_output_dimensions_pool(self):
        # TODO implement for neural network with pool layer
        self.assertTrue(True)

    def test_get_output_dimensions_activation(self):
        # TODO implement for neural network with activation layer
        self.assertTrue(True)

    def test_get_activation_function_string_relu_sigmoid(self):
        """Test case for get_activation_function_string function with relu and sigmoid function"""
        self.assertTrue(backend_utils.get_activation_function_string(self.dense_3layer_input, GCC.activation_functions) == '{2,2,1}')

    def test_get_activation_function_string_linear_tanh_softmax(self):
        """Test case for get_activation_function_string function with linear, softmax and tanh function"""
        self.dense_3layer_input['config']['layers'][0]['config']['activation'] = 'linear'
        self.dense_3layer_input['config']['layers'][1]['config']['activation'] = 'tanh'
        self.dense_3layer_input['config']['layers'][2]['config']['activation'] = 'softmax'
        self.assertTrue(backend_utils.get_activation_function_string(self.dense_3layer_input, GCC.activation_functions) == '{0,3,4}')

    def test_get_bias_information_dense3(self):
        """Test case for get_bias_information function in 3 layer dense network"""
        use_bias_string, bias_indices_string, bias_array = backend_utils.get_bias_information(self.dense_3layer_input)
        self.assertTrue(use_bias_string == '{1,1,1}'
                    and bias_indices_string == '{0,16,24}'
                    and len(bias_array) == 25
                    and backend_utils.convert_array_to_string(bias_array) == '{0.22263483703136444,0.13138847053050995,0.08629767596721649,0.5517114996910095,0.3392639756202698,-0.22105450928211212,-0.2265312224626541,-0.2264527529478073,0.8651771545410156,0.10187917202711105,-0.3911702036857605,0.2522989511489868,0.36179348826408386,-0.09673667699098587,-0.1708875298500061,-0.13006971776485443,0.31681981682777405,-0.07519398629665375,-0.19520457088947296,0.3690105378627777,-0.10645776242017746,0.09979397058486938,0.02785390242934227,0.1316162347793579,-0.2438761442899704}')

    def test_get_bias_information_dense2(self):
        """Test case for get_bias_information function in 2 layer dense network"""
        use_bias_string, bias_indices_string, bias_array = backend_utils.get_bias_information(self.dense_2layer_input)
        self.assertTrue(use_bias_string == '{1,1}'
                    and bias_indices_string == '{0,8}'
                    and len(bias_array) == 9
                    and backend_utils.convert_array_to_string(bias_array) == '{0.4447292983531952,-0.8422787189483643,0.8797550797462463,-0.8606524467468262,-0.1518070250749588,0.2564888298511505,-0.13789983093738556,-0.3959729075431824,-0.2790255844593048}')

    def test_get_bias_information_mixed(self):
        # TODO implement with mixed neural network
        self.assertTrue(True)

    def test_get_weight_information_dense3(self):
        """Test case for get_weight_information function in 3 layer dense network"""
        heights, widths = backend_utils.get_output_dimensions(self.dense_3layer_input)
        weights_indices_string, weights_array = backend_utils.get_weight_information(self.dense_3layer_input, heights)
        self.assertTrue(weights_indices_string == '{0,128,256}'
                    and len(weights_array) == 264
                    and backend_utils.convert_array_to_string(weights_array) == '{-0.8100805282592773,0.901753306388855,0.2795822024345398,-0.4192712604999542,0.6776200532913208,-0.9441443085670471,-0.2800437808036804,0.11782970279455185,-0.13435286283493042,-0.21915361285209656,0.2041853666305542,-0.7134507298469543,0.19291318953037262,-0.8826199173927307,-0.6752193570137024,0.8549830913543701,-1.3719627857208252,-0.5274502038955688,-0.23124396800994873,-0.4116051495075226,0.5376441478729248,-0.49560898542404175,0.020544910803437233,0.5829975605010986,0.7948541045188904,0.20812761783599854,1.025065302848816,-0.7788164615631104,0.7680448889732361,0.46633800864219666,-0.25286152958869934,0.30254366993904114,-0.23691658675670624,-0.36401063203811646,-0.16997449100017548,0.4575059413909912,0.8547524213790894,0.7170244455337524,-0.46226024627685547,0.6059105396270752,1.137696623802185,0.006269920151680708,0.817609429359436,0.0544731505215168,0.031075801700353622,-0.2041022628545761,0.5224971771240234,-0.009573986753821373,-0.7421486973762512,-0.23953735828399658,-0.5694900751113892,-0.2893294095993042,0.721684455871582,0.9886897802352905,-0.9591451287269592,0.03739321604371071,-0.531175434589386,0.15762042999267578,-0.1493675410747528,-0.18111944198608398,0.12199509888887405,-0.12344671040773392,0.13662931323051453,0.17638126015663147,0.08286383002996445,0.5206571817398071,-0.9635798335075378,0.41041216254234314,-0.03436516970396042,-0.5277567505836487,0.5742866396903992,0.6023487448692322,0.573002815246582,-0.1911565661430359,0.3148529827594757,-0.608443021774292,-0.674096405506134,0.15034674108028412,0.6155710816383362,-0.019679946824908257,0.26410654187202454,0.4224230647087097,0.6975041627883911,-0.003193882293999195,0.4163004755973816,0.4057024121284485,1.2035115957260132,-0.29410603642463684,-0.47574126720428467,-1.1995395421981812,-0.4154602885246277,0.032117731869220734,-0.16496165096759796,0.45419585704803467,0.4986667335033417,0.8490541577339172,-0.5362664461135864,0.5285612940788269,-0.4653119742870331,0.9023755192756653,0.24359624087810516,0.41325947642326355,-0.9940541386604309,0.6141356825828552,-0.8636071681976318,0.028472768142819405,-0.9809023141860962,-0.624100387096405,0.37142205238342285,-0.5851483345031738,0.24822849035263062,0.6492776274681091,0.6006982922554016,0.393694669008255,-0.232051819562912,-0.4690055549144745,0.214018777012825,0.1567668616771698,-1.0923436880111694,0.37128326296806335,0.2546146512031555,0.25826188921928406,-0.12362852692604065,-0.705418050289154,-0.4851866662502289,-0.5783229470252991,0.44767534732818604,-0.4371526539325714,0.43788906931877136,-0.28989502787590027,-0.8638564944267273,-0.46265366673469543,0.10203094035387039,0.648087203502655,0.7690202593803406,0.7971743941307068,-0.3299589157104492,-0.41411662101745605,0.3250950574874878,0.5065255165100098,-1.7132995128631592,0.17645160853862762,0.008479971438646317,0.081297367811203,0.23262172937393188,-1.0137851238250732,0.47564682364463806,0.3874299228191376,0.679027259349823,-0.06797883659601212,0.23567020893096924,-0.36242902278900146,0.3231452703475952,-0.8122869729995728,-0.1872439831495285,0.07921542972326279,-0.519558310508728,0.4604364037513733,-0.24832335114479065,0.4852125346660614,0.08519076555967331,-0.16626712679862976,0.2962987720966339,-0.010394229553639889,-0.5785297155380249,0.4305243492126465,-1.4546102285385132,0.18866503238677979,0.42650434374809265,-1.051063895225525,0.04086971655488014,0.8141140341758728,-0.08182719349861145,-0.8611471056938171,0.4419907033443451,-0.4531274437904358,0.6173373460769653,-0.966194212436676,0.6517370939254761,-0.5217079520225525,0.1584899127483368,-0.7305968999862671,0.02816818282008171,0.8957902193069458,-0.8196432590484619,0.3166002631187439,-0.32218068838119507,-0.17739717662334442,0.6111621260643005,-1.162420392036438,0.1945500671863556,0.49178507924079895,0.7994028925895691,0.46749967336654663,-0.42953523993492126,-0.02985234372317791,0.9004541635513306,-0.5809367895126343,0.4783508777618408,0.24079208076000214,0.5156621336936951,-0.2139977663755417,0.5525525212287903,0.2591993510723114,-0.3627344071865082,-0.03227323293685913,-0.39246830344200134,0.6430918574333191,0.806722104549408,0.6563072800636292,0.3768554925918579,-0.2531486451625824,-0.7142532467842102,-0.289291113615036,-1.8243343830108643,0.004458898678421974,0.5696000456809998,-0.4135691225528717,-0.44411909580230713,-0.15006788074970245,-0.4105619192123413,-0.02028571628034115,-0.1990935057401657,0.5832392573356628,-0.476336807012558,-0.21040119230747223,-0.10455772280693054,0.38163045048713684,0.13246354460716248,-1.0955878496170044,0.29251164197921753,-0.5789858102798462,0.22669632732868195,0.17539441585540771,0.22307905554771423,-0.10287618637084961,0.49935826659202576,-2.2194831371307373,0.7326659560203552,0.33501294255256653,-0.09763967245817184,0.6550970673561096,-0.08206404000520706,0.42008838057518005,0.3385299742221832,-1.3764796257019043,0.4596630036830902,-0.08332288265228271,0.19659259915351868,0.6042377352714539,0.2704513669013977,-0.512715220451355,0.7208464741706848,-0.12545344233512878,-0.06954788416624069,-0.8529912829399109,-0.7719011902809143,1.802193522453308,0.973081111907959,0.4021240174770355,1.880678653717041,-3.2501473426818848,1.3993031978607178,-0.8899591565132141}')

    def test_get_weight_information_dense2(self):
        """Test case for get_weight_information function in 2 layer dense network"""
        heights, widths = backend_utils.get_output_dimensions(self.dense_2layer_input)
        weights_indices_string, weights_array = backend_utils.get_weight_information(self.dense_2layer_input, heights)
        self.assertTrue(weights_indices_string == '{0,64}'
                    and len(weights_array) == 72
                    and backend_utils.convert_array_to_string(weights_array) == '{-0.15035194158554077,1.430967926979065,0.5391924381256104,-0.3918966054916382,0.4462871849536896,0.5552943348884583,0.7700048089027405,0.28981146216392517,-1.689786672592163,1.042858362197876,2.2847230434417725,-1.9677045345306396,-0.06611120700836182,-0.061966150999069214,-0.6471585631370544,1.4964995384216309,0.19608011841773987,-0.6369050741195679,0.09111300855875015,0.13803385198116302,-0.3828596770763397,-0.4359494745731354,0.7322076559066772,-0.049885910004377365,0.4382251799106598,0.7386993169784546,-0.7968388199806213,0.40169671177864075,1.0121262073516846,0.2377474009990692,0.4582376778125763,-0.26966771483421326,0.1630777269601822,-0.869050920009613,-0.4613328278064728,0.2624562084674835,-0.9985146522521973,-0.6456401944160461,-0.12137659639120102,1.0982298851013184,-0.03326864913105965,0.6471387147903442,-0.1343340277671814,0.22081176936626434,0.854404628276825,-1.670728087425232,-0.16672520339488983,0.8519517779350281,1.2319378852844238,-0.01635606400668621,0.28598904609680176,-0.34935298562049866,0.049734167754650116,-1.2383219003677368,-0.7718221545219421,0.7590272426605225,0.3540281057357788,-1.3175017833709717,1.9483529329299927,-2.1833057403564453,-1.0394269227981567,-0.08081339299678802,-2.6470179557800293,-0.5022009611129761,-0.8252769708633423,1.2357004880905151,1.0166884660720825,-1.3550186157226562,1.1026166677474976,-1.6992595195770264,-0.9769604206085205,0.8309341073036194}')

    def test_get_weight_information_mixed(self):
        # TODO implement with mixed neural network
        self.assertTrue(True)

#? ############### INFO ###############
#? This script has to be run with -m switch from parent directory in order to get the imports right
#? Directory: Neural-Network-Translator (repository root directory)
#? Command: python -m test.backend_utils_test

if __name__ == '__main__':
    #? Searching for all test cases in TestBackendUtils
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestBackendUtils)

    #? Adding all found test cases
    suite = unittest.TestSuite()
    for test_name in test_names:
        suite.addTest(TestBackendUtils(test_name))

    #? Running the test suite
    result = unittest.TextTestRunner().run(suite)
    sys.exit(not result.wasSuccessful())