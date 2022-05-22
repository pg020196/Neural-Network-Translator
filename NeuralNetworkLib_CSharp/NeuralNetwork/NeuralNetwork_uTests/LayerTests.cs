//using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using Tensor;
using Layers;
using NUnit.Framework;


namespace NeuralNetworkTests
{
    [TestFixture]
    public class TestLayer
    {
        [Test]
        public void InputLayer_CheckResultShapeAndValues()
        {
            // Arrage
            int[] inputShape = { 100, 100, 3 };
            int batchSize = 32;
            int[] batchShape = { batchSize, 100, 100, 3 };
            var input = Tensor<double>.randUniform(0, 1, batchShape);

            // Act
            var inputLayer = new InputLayer<double>(inputShape);
            var output = inputLayer.FeedForward(input);

            // Assert
            CollectionAssert.AreEqual(batchShape, output.Shape, "Failed to produce output tensor of correct shape");
            var tmp0 = input.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < input.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void InputLayer_0DInput_ShouldThrowArgumentException()
        {
            // Arrage
            int[] inputShape = { 100, 100, 3 };
            int[] batchShape = { 10 }; // batchShape should have 1 dimension more than inputShape --> Exception
            var input = Tensor<double>.randUniform(0, 1, batchShape);

            // Act
            var inputLayer = new InputLayer<double>(inputShape);

            // Assert
            Assert.Throws<ArgumentException>(() => inputLayer.FeedForward(input));
        }


        [Test]
        public void InputLayer_WrongShape_ShouldThrowArgumentException()
        {
            // Arrage
            int[] inputShape = { 100, 100, 3 };
            int[] batchShape = inputShape; // batchShape should have 1 dimension more than inputShape --> Exception
            var input = Tensor<double>.randUniform(0, 1, batchShape);

            // Act
            var inputLayer = new InputLayer<double>(inputShape);

            // Assert
            Assert.Throws<ArgumentException>(() => inputLayer.FeedForward(input));
        }

        [Test]
        public void DenseLayer_CheckResultShapeAndValues()
        {
            // Arrange
            int numUnits = 12;
            int[] layerInputShape = { 10 };
            int batchSize = 8;
            ActivationType activation = ActivationType.tanh;
            int[] inputShape = { batchSize, layerInputShape[0] };
            var input = new Tensor<double>(new double[] { 0.80681314, 0.7997264 , 0.70624335, 0.89483646, 0.07636926,
                                                          0.11436052, 0.05661003, 0.75986455, 0.99628727, 0.86914269,
                                                          0.87800169, 0.41818808, 0.74435962, 0.04925722, 0.53935476,
                                                          0.08602625, 0.26448075, 0.68999073, 0.8706262 , 0.59878295,
                                                          0.64562871, 0.31374657, 0.15357318, 0.24123003, 0.86438314,
                                                          0.61936406, 0.35747239, 0.29098033, 0.88851188, 0.9388363 ,
                                                          0.46196257, 0.32244808, 0.76202514, 0.79200692, 0.19512262,
                                                          0.02075605, 0.48444832, 0.89831978, 0.97664915, 0.37386355,
                                                          0.27378221, 0.24924764, 0.78837814, 0.64292364, 0.15555527,
                                                          0.38186777, 0.10715792, 0.77501141, 0.51600529, 0.92052122,
                                                          0.30267189, 0.88052957, 0.29841503, 0.19447701, 0.75296191,
                                                          0.37519068, 0.23929142, 0.05825534, 0.22087956, 0.14476802,
                                                          0.83470113, 0.03705189, 0.70091882, 0.14622499, 0.38624479,
                                                          0.72497712, 0.21980499, 0.14993275, 0.85286238, 0.66552102,
                                                          0.80191709, 0.65449895, 0.4396946 , 0.55231779, 0.32443396,
                                                          0.92377668, 0.03723244, 0.8051447 , 0.11779034, 0.01306947});
            input = input.reshape(inputShape);
            int[] weightsShape = { layerInputShape[0], numUnits };
            var weights = new Tensor<double>(new double[] { -0.73237871,  0.83062599,  0.27572447, -0.43370734,  0.09517949,
                                                            -0.07957397,  0.47046668,  1.71429601,  1.48539842, -0.45659543,
                                                             1.01028974,  0.19984724, -1.06676089, -0.17750272,  0.45227792,
                                                            -1.90492634, -0.68424801, -1.31890318, -0.28752037,  0.63163731,
                                                             0.65010523,  0.76832687, -0.38125513,  0.03710796, -1.35178256,
                                                            -0.61867191,  0.46279407,  0.04863851, -1.32251246,  0.46560482,
                                                             0.34465054, -1.02005927, -1.12728475, -0.89936221, -0.78850199,
                                                            -0.35206277, -0.97418917,  0.64762491, -0.42112665, -0.97496955,
                                                             0.61377813,  2.1395292 , -0.74532113,  0.85852631, -1.75907088,
                                                            -0.71561713,  1.62823035, -0.56184622, -1.251919  ,  1.01018933,
                                                             0.10736353, -0.28425875, -1.34923182, -1.30914552,  1.15607108,
                                                            -0.26650163,  0.65315548, -0.37046224, -0.14848824, -0.3828908 ,
                                                            -0.27077534, -0.9541994 ,  0.39505703, -0.19246699,  1.12980992,
                                                            -2.16576039, -0.18932236,  0.60782956, -0.70430811, -0.25852463,
                                                            -0.61770276,  0.39280668, -0.03698243,  0.39005703, -0.37550031,
                                                             0.29026368, -1.41724596,  1.24351068, -0.58896579, -0.45254066,
                                                             1.13490479, -1.45041859,  1.00104014,  0.33329005, -0.49995619,
                                                            -0.08033119,  1.33983489,  0.37647476,  0.16203691,  0.36936748,
                                                            -0.10581103,  0.79330457,  0.78686027,  1.28361513, -0.49344423,
                                                            -0.64874544, -1.67993226, -2.00819325, -0.63096792, -0.31769837,
                                                             0.48036299,  0.955223  , -0.98087482, -0.27788969,  1.51170291,
                                                            -0.60417621,  0.52123612, -0.31815622, -0.36910681, -0.43517113,
                                                             0.75083538,  1.25875907, -1.35971916,  0.18227603, -0.10230348,
                                                             0.21008403, -0.74033765, -0.75393196,  1.68344422, -0.72761516});
            weights = weights.reshape(weightsShape);
            int[] biasShape = { numUnits };
            var bias = new Tensor<double>(new double[] { -0.45522274, -2.01156394,  0.40782968,  0.53340629, -0.30200196,
                                                          1.57879627, -1.29852824, -0.00239608,  1.41920653,  0.19661919,
                                                          1.06622205,  1.01150839});
            bias = bias.reshape(biasShape);
            int[] expectedOuputShape = { 8, 12 };
            var expectedOutput = new Tensor<double>(new double[]{-0.99999222, -0.99898077,  0.96515197, -0.81499153, -0.94590334,
                                                                  0.99903245, -0.99074732,  0.98567255,  0.97851848, -0.84958095,
                                                                  0.99940364, -0.74283063, -0.99995358, -0.99719236,  0.96259423,
                                                                 -0.01203947, -0.99183182,  0.96684328, -0.86743641,  0.85379526,
                                                                  0.99942513, -0.85783633,  0.98763638, -0.32042808, -0.99990027,
                                                                 -0.99708256,  0.89261382,  0.17409661, -0.97953637,  0.66409215,
                                                                 -0.91190485,  0.90844027,  0.99775641, -0.96660407,  0.9985592 ,
                                                                 -0.15916012, -0.99996195, -0.99807878,  0.88865218, -0.41084914,
                                                                 -0.94887613,  0.99986482, -0.99019437,  0.82006929,  0.98970239,
                                                                 -0.90831682,  0.99691895, -0.58373094, -0.99975803, -0.99839757,
                                                                  0.97674939,  0.50151104, -0.95649606,  0.99525227, -0.96967022,
                                                                  0.82717772,  0.55866034, -0.84221203,  0.99461674, -0.64918986,
                                                                 -0.99881615, -0.96334365,  0.81181281, -0.91095478, -0.97392747,
                                                                 -0.28530052, -0.77411633,  0.70114843,  0.98749931, -0.51051781,
                                                                  0.87256429,  0.54817625, -0.99981248, -0.99917095,  0.86810305,
                                                                  0.40534906, -0.91696865,  0.85955201, -0.90949272,  0.80843824,
                                                                  0.98767318, -0.98250006,  0.99243191,  0.20511163, -0.99948906,
                                                                 -0.97677949,  0.97993391, -0.91466172, -0.19117276,  0.07124812,
                                                                 -0.88390951,  0.99369654,  0.96215138,  0.08192397,  0.85755799,
                                                                  0.39073868 });
            expectedOutput = expectedOutput.reshape(expectedOuputShape);

            // Act
            var dense = new Dense<double>(layerInputShape, numUnits, activation);
            dense.Weights = weights;
            dense.Bias = bias;
            var output = dense.FeedForward(input);
            var actualOutputShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOuputShape, actualOutputShape, "Failed to produce output tensor with correct shape");
            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < expectedOutput.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void PoolingLayer1D_AvgPoolingUnitStridePaddingValid_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 10, 3 };
            int batchSize = 5;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1] };
            var input = new Tensor<double>(new double[] {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                                           13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                                                           26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                                                           39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                                                           52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                                                           65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                                                           78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                                                           91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
                                                          104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                                                          117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                                                          130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                                                          143, 144, 145, 146, 147, 148, 149});
            input = input.reshape(inputShape);

            int pool_size = 3;
            int stride = 1;
            PaddingType paddingType = PaddingType.valid;
            PoolingType poolingType = PoolingType.average;

            int[] expectedOutputShape = { batchSize, layerInputShape[0] - pool_size + 1, layerInputShape[1] };
            var expectedOutput = new Tensor<double>(new double[] {  3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                                                                   14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
                                                                   25,  26,  33,  34,  35,  36,  37,  38,  39,  40,  41,
                                                                   42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
                                                                   53,  54,  55,  56,  63,  64,  65,  66,  67,  68,  69,
                                                                   70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                                                                   81,  82,  83,  84,  85,  86,  93,  94,  95,  96,  97,
                                                                   98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                                                                  109, 110, 111, 112, 113, 114, 115, 116, 123, 124, 125,
                                                                  126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
                                                                  137, 138, 139, 140, 141, 142, 143, 144, 145, 146});
            expectedOutput = expectedOutput.reshape(expectedOutputShape);


            // Act
            var poolingLayer = new PoolingLayer1D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output of correct shape");

            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output with correct values");
        }

        [Test]
        public void PoolingLayer1D_MaxPoolingUnitStridePaddingValid_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 10, 3 };
            int batchSize = 5;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1] };
            var input = new Tensor<double>(new double[] {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                                           13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                                                           26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                                                           39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                                                           52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                                                           65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                                                           78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                                                           91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
                                                          104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                                                          117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                                                          130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                                                          143, 144, 145, 146, 147, 148, 149});
            input = input.reshape(inputShape);

            int pool_size = 3;
            int stride = 1;
            PaddingType paddingType = PaddingType.valid;
            PoolingType poolingType = PoolingType.max;

            int[] expectedOutputShape = { batchSize, layerInputShape[0] - pool_size + 1, layerInputShape[1] };
            var expectedOutput = new Tensor<double>(new double[] {   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,
                                                                    19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  36,  37,
                                                                    38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
                                                                    51,  52,  53,  54,  55,  56,  57,  58,  59,  66,  67,  68,  69,
                                                                    70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,
                                                                    83,  84,  85,  86,  87,  88,  89,  96,  97,  98,  99, 100, 101,
                                                                   102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                                                                   115, 116, 117, 118, 119, 126, 127, 128, 129, 130, 131, 132, 133,
                                                                   134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
                                                                   147, 148, 149});
            expectedOutput = expectedOutput.reshape(expectedOutputShape);


            // Act
            var poolingLayer = new PoolingLayer1D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output of correct shape");

            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output with correct values");
        }


        [Test]
        public void PoolingLayer1D_AvgPoolingUnitStridePaddingSame_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 10, 3 };
            int batchSize = 5;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1] };
            var input = new Tensor<double>(new double[] {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                                           13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                                                           26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                                                           39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                                                           52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                                                           65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                                                           78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                                                           91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
                                                          104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                                                          117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                                                          130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                                                          143, 144, 145, 146, 147, 148, 149});
            input = input.reshape(inputShape);

            int pool_size = 3;
            int stride = 1;
            PaddingType paddingType = PaddingType.same_keras;
            PoolingType poolingType = PoolingType.average;

            int[] expectedOutputShape = { batchSize, (int)Math.Ceiling((double)(layerInputShape[0] / stride)), layerInputShape[1] };
            var expectedOutput = new Tensor<double>(new double[] {     1.5,   2.5,   3.5,   3  ,   4  ,   5  ,   6  ,   7  ,   8  ,
                                                                       9  ,  10  ,  11  ,  12  ,  13  ,  14  ,  15  ,  16  ,  17  ,
                                                                      18  ,  19  ,  20  ,  21  ,  22  ,  23  ,  24  ,  25  ,  26  ,
                                                                      25.5,  26.5,  27.5,  31.5,  32.5,  33.5,  33  ,  34  ,  35  ,
                                                                      36  ,  37  ,  38  ,  39  ,  40  ,  41  ,  42  ,  43  ,  44  ,
                                                                      45  ,  46  ,  47  ,  48  ,  49  ,  50  ,  51  ,  52  ,  53  ,
                                                                      54  ,  55  ,  56  ,  55.5,  56.5,  57.5,  61.5,  62.5,  63.5,
                                                                      63  ,  64  ,  65  ,  66  ,  67  ,  68  ,  69  ,  70  ,  71  ,
                                                                      72  ,  73  ,  74  ,  75  ,  76  ,  77  ,  78  ,  79  ,  80  ,
                                                                      81  ,  82  ,  83  ,  84  ,  85  ,  86  ,  85.5,  86.5,  87.5,
                                                                      91.5,  92.5,  93.5,  93  ,  94  ,  95  ,  96  ,  97  ,  98  ,
                                                                      99  , 100  , 101  , 102  , 103  , 104  , 105  , 106  , 107  ,
                                                                     108  , 109  , 110  , 111  , 112  , 113  , 114  , 115  , 116  ,
                                                                     115.5, 116.5, 117.5, 121.5, 122.5, 123.5, 123  , 124  , 125  ,
                                                                     126  , 127  , 128  , 129  , 130  , 131  , 132  , 133  , 134  ,
                                                                     135  , 136  , 137  , 138  , 139  , 140  , 141  , 142  , 143  ,
                                                                     144  , 145  , 146  , 145.5, 146.5, 147.5});
            expectedOutput = expectedOutput.reshape(expectedOutputShape);


            // Act
            var poolingLayer = new PoolingLayer1D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output of correct shape");

            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output with correct values");
        }

        [Test]
        public void PoolingLayer1D_AvgPoolingNonUnitStridePaddingSame_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 10, 3 };
            int batchSize = 5;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1] };
            var input = new Tensor<double>(new double[] {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                                           13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                                                           26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                                                           39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                                                           52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                                                           65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                                                           78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                                                           91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
                                                          104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                                                          117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                                                          130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                                                          143, 144, 145, 146, 147, 148, 149});
            input = input.reshape(inputShape);

            int pool_size = 3;
            int stride = 2;
            PaddingType paddingType = PaddingType.same_keras;
            PoolingType poolingType = PoolingType.average;

            int[] expectedOutputShape = { batchSize, (int)Math.Ceiling((double)(layerInputShape[0] / stride)), layerInputShape[1] };
            var expectedOutput = new Tensor<double>(new double[] {  3  ,   4  ,   5  ,   9  ,  10  ,  11  ,  15  ,  16  ,  17  ,
                                                                   21  ,  22  ,  23  ,  25.5,  26.5,  27.5,  33  ,  34  ,  35  ,
                                                                   39  ,  40  ,  41  ,  45  ,  46  ,  47  ,  51  ,  52  ,  53  ,
                                                                   55.5,  56.5,  57.5,  63  ,  64  ,  65  ,  69  ,  70  ,  71  ,
                                                                   75  ,  76  ,  77  ,  81  ,  82  ,  83  ,  85.5,  86.5,  87.5,
                                                                   93  ,  94  ,  95  ,  99  , 100  , 101  , 105  , 106  , 107  ,
                                                                  111  , 112  , 113  , 115.5, 116.5, 117.5, 123  , 124  , 125  ,
                                                                  129  , 130  , 131  , 135  , 136  , 137  , 141  , 142  , 143  ,
                                                                  145.5, 146.5, 147.5});
            expectedOutput = expectedOutput.reshape(expectedOutputShape);

            // Act
            var poolingLayer = new PoolingLayer1D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output of correct shape");

            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output with correct values");
        }

        [Test]
        public void PoolingLayer2D_AvgPoolingUnitStridePaddingValid_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 7, 7, 3 };
            int batchSize = 2;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1], layerInputShape[2] };
            var input = new Tensor<double>(new double[] { 0.56146491, -0.4325621 ,  0.88601032, -0.17517148,  1.38847498,
                                                         -0.08028661, -1.12155469, -0.58778088, -0.46272179, -0.52845978,
                                                          1.30353991, -1.24046951,  0.17010402,  1.67470356, -0.20598032,
                                                         -0.13742261,  0.54233763,  0.36579914, -1.52907043, -0.14506615,
                                                          0.40729955,  0.35551977,  0.86400829,  1.88329039,  0.29745863,
                                                         -0.27324774, -0.73431326,  1.3180166 , -0.17401557, -0.10312114,
                                                         -0.69645137, -1.56197892,  0.42986977, -0.60144652, -0.04465733,
                                                          0.11235045,  0.95659649,  0.11681149, -2.13546254, -0.22732446,
                                                         -0.69928343, -1.46921705, -1.44556146, -1.153496  ,  0.09489746,
                                                         -2.12854295,  1.47798476, -0.20997356,  1.97453884,  1.30407624,
                                                         -0.19798062,  1.40357332,  0.26905054,  0.90630356,  0.89581737,
                                                         -0.31213708,  2.37562488, -0.12294524,  0.46848971, -0.08462803,
                                                          0.61212434, -0.70103499,  1.17807189, -1.38645167, -0.85455524,
                                                         -0.89186198,  0.57573523, -0.96622848,  2.98030596, -2.66874266,
                                                          0.46382983,  0.72423461,  1.55503865,  1.01333431,  1.16964474,
                                                          0.95433798, -0.41045219, -1.17181812, -1.68892746,  0.56143387,
                                                          1.49089948, -0.63263902, -1.20353446,  0.10338993,  0.01262961,
                                                          0.94779568, -1.87370706,  0.06609356,  0.74285845,  0.66433993,
                                                          2.20110966, -0.57995003, -0.98022386,  0.81195219, -0.33806267,
                                                         -0.47705876, -0.04729528,  0.08869091,  0.85609904,  1.18131153,
                                                          0.32612134,  1.12162879, -0.13882688, -1.83888855,  0.05444741,
                                                          1.59519607,  0.52501207,  1.1060692 ,  0.85094447,  1.24380913,
                                                          1.57511566,  0.7152266 ,  0.35566718,  0.42239772,  1.15564331,
                                                         -0.23656388, -1.28050578, -1.65900638, -0.09968587, -0.46287323,
                                                          1.00145748, -0.56657201,  0.43335275, -0.4575495 , -0.50672402,
                                                          1.66693205,  0.18318202, -0.40510412, -2.62397045, -0.66257671,
                                                         -1.77384248,  1.50004202, -0.63067836, -0.79536606, -2.3626925 ,
                                                         -1.05587481, -0.16627958,  3.15725458, -0.65810233, -0.06300574,
                                                         -1.91743821,  2.09829067,  0.63034419, -0.0798598 , -0.52319696,
                                                          0.07392963,  0.49856004,  0.21049685, -0.08628767, -1.12611707,
                                                         -2.03317302,  0.20303325,  0.59996247, -0.5869878 ,  0.4868331 ,
                                                          0.02867377,  0.94264783,  0.69660946, -0.00520433, -0.27734673,
                                                         -0.72081539, -0.13158885,  0.9802476 ,  1.29261004,  0.50604571,
                                                          0.77498456, -0.01284429, -0.08408896, -0.60987957, -0.00892505,
                                                         -0.99751832,  0.2806547 , -1.40819646,  0.31895987,  1.17092962,
                                                         -0.91396889, -1.30201044,  0.08038122, -1.00798243, -0.10354281,
                                                         -0.25034687,  1.4214869 ,  0.05908077,  0.92222859,  0.22306133,
                                                          0.59398505, -0.29965699,  0.86203472,  0.27997115, -1.59815247,
                                                          0.90234873,  0.08397274,  2.15504169, -1.51010251,  1.74955757,
                                                          0.82895949, -2.03828956, -1.40329323, -1.02550191,  1.15027985,
                                                         -0.91570947, -1.96525968, -0.95385491, -0.75978529,  0.53881717,
                                                         -0.29598287, -0.18880006,  0.39676459, -1.50394917,  0.41602566,
                                                         -0.26422785,  0.66200554, -0.18655683,  1.64482807,  0.25808603,
                                                         -2.76706006,  0.19007231, -1.38818591, -0.07925095, -2.99019342,
                                                         -0.18778626,  0.04614819, -0.00344016,  0.39003154, -1.34984968,
                                                         -0.18631236,  1.24700716, -0.31427149,  0.45359917,  0.26096725,
                                                         -0.62056391, -1.19134459,  0.85099639,  0.64249793,  0.03684947,
                                                         -0.65883196,  0.10765061, -0.92412945,  0.60890281, -1.35575801,
                                                         -0.35015551,  0.92508287, -1.10579289, -1.17649377,  1.15308594,
                                                          0.20670113, -0.60592161, -1.06516026, -0.39640956, -0.74377697,
                                                         -0.37722864, -0.69816568,  2.63162303, -2.13000723, -1.84351966,
                                                          0.90317835, -0.1214262 ,  0.06524366, -0.02012757, -0.63654065,
                                                         -0.92797039,  1.45051906, -1.95582026,  1.26315388, -0.43424369,
                                                          1.27432657,  0.72740755,  0.63033444,  1.65289248,  0.91875244,
                                                         -1.84788191, -0.65909792, -0.32767424,  1.23565268,  0.35926131,
                                                         -0.22272   , -0.98385977, -0.75728687, -0.90239033, -0.20538704,
                                                          0.33745069,  0.33083082,  1.11875784,  0.87167551,  1.32079975,
                                                          0.97008268, -0.50337526, -1.24629487, -0.11255154,  0.07906763,
                                                         -0.54813318,  1.39203082,  0.19450113,  0.20825211});
            input = input.reshape(inputShape);

            int[] pool_size = { 3, 3 };
            int[] stride = { 1, 1 };
            PaddingType paddingType = PaddingType.valid;
            PoolingType poolingType = PoolingType.average;

            int[] expectedOutputShape = { batchSize, layerInputShape[0]-pool_size[0]+1, layerInputShape[1]-pool_size[1]+1, layerInputShape[2]};
            var expectedOutput = new Tensor<double>(new double[] {-4.04257514e-02,  2.68160224e-01,  1.19533472e-01,  3.81563492e-02,
                                                                   3.49567056e-01, -1.88077003e-01,  3.12682003e-01,  2.07866728e-01,
                                                                   1.79319486e-01,  1.48818418e-01,  2.72906601e-01,  5.81563897e-02,
                                                                   1.82588235e-03,  1.00018166e-01,  6.04286641e-02, -3.45336616e-01,
                                                                   7.64840096e-02,  3.93941998e-01,  1.81180507e-01,  1.72533855e-01,
                                                                   5.51663339e-01,  4.59409118e-01,  6.07833080e-02,  4.71678674e-01,
                                                                   2.95065969e-01,  1.10993646e-02,  3.43642682e-01,  1.61770377e-02,
                                                                  -2.47151583e-01,  4.43567634e-02, -3.11021358e-01,  1.53590575e-01,
                                                                   3.44478898e-02,  4.21195090e-01,  3.76321435e-01,  5.08843601e-01,
                                                                   7.86703348e-01,  1.66486636e-01,  3.56091708e-01,  5.49206972e-01,
                                                                   1.85163185e-01,  6.87410593e-01,  1.12550825e-01, -3.35701287e-01,
                                                                   6.58190489e-01,  2.17971161e-01,  2.08693191e-01,  4.14074481e-01,
                                                                   5.84777892e-01,  1.88743755e-01,  5.33138931e-01,  3.35362673e-01,
                                                                   2.85341702e-02, -1.33344844e-01,  3.62723559e-01,  3.75826359e-02,
                                                                   1.86596543e-01, -1.65237531e-01, -4.05512333e-01,  4.54673141e-01,
                                                                   4.81236398e-01,  2.89866254e-02, -2.85847723e-01,  3.83537769e-01,
                                                                  -1.71969980e-01,  2.46518776e-01,  9.25527364e-02, -2.03839511e-01,
                                                                  -3.38337868e-01,  3.14264029e-01, -4.72236946e-02,  1.50066599e-01,
                                                                   8.85647014e-02, -2.17310011e-01,  2.41205424e-01, -2.02344786e-02,
                                                                  -4.85950589e-01, -2.27534741e-01,  2.01439083e-01, -4.82420444e-01,
                                                                  -1.14734069e-01, -1.20280534e-01, -2.08855733e-01, -5.03708899e-01,
                                                                  -6.01475779e-03,  2.00601310e-01, -1.05057701e-01,  9.11591351e-02,
                                                                   3.46385092e-02,  7.67605752e-02,  4.22025084e-01, -6.05025351e-01,
                                                                  -5.09244382e-01,  2.59463578e-01, -7.82905042e-01, -4.95133460e-01,
                                                                  -4.40488815e-01, -3.92029971e-01, -6.45356953e-01, -5.42180777e-01,
                                                                   2.20695585e-01, -3.25860560e-01, -4.37340625e-02,  1.83422446e-01,
                                                                  -2.09356442e-01,  9.75440815e-02, -2.57007807e-01, -3.56471121e-01,
                                                                  -4.82476912e-02, -3.15649390e-01, -6.35945380e-01, -8.24015856e-01,
                                                                  -3.78592797e-02, -7.46287763e-01, -8.62717867e-01,  2.62522578e-01,
                                                                  -5.30863225e-01, -3.65780383e-01, -1.27231553e-01, -4.11679894e-01,
                                                                   3.34080189e-01, -2.83889025e-01, -7.04969287e-01, -6.57318719e-03,
                                                                  -3.50724369e-01, -5.28181791e-01, -4.73132461e-01,  2.03440860e-02,
                                                                  -2.86134601e-01, -4.07323033e-01,  3.81517768e-01, -4.62267140e-04,
                                                                  -4.34904099e-01,  4.30758238e-01, -2.06008181e-01,  1.64717242e-01,
                                                                  -2.38609076e-01, -4.56237257e-01,  1.13960609e-01, -1.54088199e-01,
                                                                  -1.33803651e-01,  4.76469658e-02,  2.30532005e-01, -8.74358788e-02,
                                                                   1.65592015e-01,  2.70197272e-01,  1.26687124e-01, -2.14269072e-01,
                                                                   1.94334626e-01, -1.28396049e-01});
            
            expectedOutput = expectedOutput.reshape(expectedOutputShape);

            // Act
            var poolingLayer = new PoolingLayer2D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output tensor with correct shape");
            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-7, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void PoolingLayer2D_MaxPoolingUnitStridePaddingValid_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 7, 7, 3 };
            int batchSize = 2;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1], layerInputShape[2] };
            var input = new Tensor<double>(new double[] { 0.56146491, -0.4325621 ,  0.88601032, -0.17517148,  1.38847498,
                                                         -0.08028661, -1.12155469, -0.58778088, -0.46272179, -0.52845978,
                                                          1.30353991, -1.24046951,  0.17010402,  1.67470356, -0.20598032,
                                                         -0.13742261,  0.54233763,  0.36579914, -1.52907043, -0.14506615,
                                                          0.40729955,  0.35551977,  0.86400829,  1.88329039,  0.29745863,
                                                         -0.27324774, -0.73431326,  1.3180166 , -0.17401557, -0.10312114,
                                                         -0.69645137, -1.56197892,  0.42986977, -0.60144652, -0.04465733,
                                                          0.11235045,  0.95659649,  0.11681149, -2.13546254, -0.22732446,
                                                         -0.69928343, -1.46921705, -1.44556146, -1.153496  ,  0.09489746,
                                                         -2.12854295,  1.47798476, -0.20997356,  1.97453884,  1.30407624,
                                                         -0.19798062,  1.40357332,  0.26905054,  0.90630356,  0.89581737,
                                                         -0.31213708,  2.37562488, -0.12294524,  0.46848971, -0.08462803,
                                                          0.61212434, -0.70103499,  1.17807189, -1.38645167, -0.85455524,
                                                         -0.89186198,  0.57573523, -0.96622848,  2.98030596, -2.66874266,
                                                          0.46382983,  0.72423461,  1.55503865,  1.01333431,  1.16964474,
                                                          0.95433798, -0.41045219, -1.17181812, -1.68892746,  0.56143387,
                                                          1.49089948, -0.63263902, -1.20353446,  0.10338993,  0.01262961,
                                                          0.94779568, -1.87370706,  0.06609356,  0.74285845,  0.66433993,
                                                          2.20110966, -0.57995003, -0.98022386,  0.81195219, -0.33806267,
                                                         -0.47705876, -0.04729528,  0.08869091,  0.85609904,  1.18131153,
                                                          0.32612134,  1.12162879, -0.13882688, -1.83888855,  0.05444741,
                                                          1.59519607,  0.52501207,  1.1060692 ,  0.85094447,  1.24380913,
                                                          1.57511566,  0.7152266 ,  0.35566718,  0.42239772,  1.15564331,
                                                         -0.23656388, -1.28050578, -1.65900638, -0.09968587, -0.46287323,
                                                          1.00145748, -0.56657201,  0.43335275, -0.4575495 , -0.50672402,
                                                          1.66693205,  0.18318202, -0.40510412, -2.62397045, -0.66257671,
                                                         -1.77384248,  1.50004202, -0.63067836, -0.79536606, -2.3626925 ,
                                                         -1.05587481, -0.16627958,  3.15725458, -0.65810233, -0.06300574,
                                                         -1.91743821,  2.09829067,  0.63034419, -0.0798598 , -0.52319696,
                                                          0.07392963,  0.49856004,  0.21049685, -0.08628767, -1.12611707,
                                                         -2.03317302,  0.20303325,  0.59996247, -0.5869878 ,  0.4868331 ,
                                                          0.02867377,  0.94264783,  0.69660946, -0.00520433, -0.27734673,
                                                         -0.72081539, -0.13158885,  0.9802476 ,  1.29261004,  0.50604571,
                                                          0.77498456, -0.01284429, -0.08408896, -0.60987957, -0.00892505,
                                                         -0.99751832,  0.2806547 , -1.40819646,  0.31895987,  1.17092962,
                                                         -0.91396889, -1.30201044,  0.08038122, -1.00798243, -0.10354281,
                                                         -0.25034687,  1.4214869 ,  0.05908077,  0.92222859,  0.22306133,
                                                          0.59398505, -0.29965699,  0.86203472,  0.27997115, -1.59815247,
                                                          0.90234873,  0.08397274,  2.15504169, -1.51010251,  1.74955757,
                                                          0.82895949, -2.03828956, -1.40329323, -1.02550191,  1.15027985,
                                                         -0.91570947, -1.96525968, -0.95385491, -0.75978529,  0.53881717,
                                                         -0.29598287, -0.18880006,  0.39676459, -1.50394917,  0.41602566,
                                                         -0.26422785,  0.66200554, -0.18655683,  1.64482807,  0.25808603,
                                                         -2.76706006,  0.19007231, -1.38818591, -0.07925095, -2.99019342,
                                                         -0.18778626,  0.04614819, -0.00344016,  0.39003154, -1.34984968,
                                                         -0.18631236,  1.24700716, -0.31427149,  0.45359917,  0.26096725,
                                                         -0.62056391, -1.19134459,  0.85099639,  0.64249793,  0.03684947,
                                                         -0.65883196,  0.10765061, -0.92412945,  0.60890281, -1.35575801,
                                                         -0.35015551,  0.92508287, -1.10579289, -1.17649377,  1.15308594,
                                                          0.20670113, -0.60592161, -1.06516026, -0.39640956, -0.74377697,
                                                         -0.37722864, -0.69816568,  2.63162303, -2.13000723, -1.84351966,
                                                          0.90317835, -0.1214262 ,  0.06524366, -0.02012757, -0.63654065,
                                                         -0.92797039,  1.45051906, -1.95582026,  1.26315388, -0.43424369,
                                                          1.27432657,  0.72740755,  0.63033444,  1.65289248,  0.91875244,
                                                         -1.84788191, -0.65909792, -0.32767424,  1.23565268,  0.35926131,
                                                         -0.22272   , -0.98385977, -0.75728687, -0.90239033, -0.20538704,
                                                          0.33745069,  0.33083082,  1.11875784,  0.87167551,  1.32079975,
                                                          0.97008268, -0.50337526, -1.24629487, -0.11255154,  0.07906763,
                                                         -0.54813318,  1.39203082,  0.19450113,  0.20825211});
            input = input.reshape(inputShape);

            int[] pool_size = { 3, 3 };
            int[] stride = { 1, 1 };
            PaddingType paddingType = PaddingType.valid;
            PoolingType poolingType = PoolingType.max;

            int[] expectedOutputShape = { batchSize, layerInputShape[0] - pool_size[0] + 1, layerInputShape[1] - pool_size[1] + 1, layerInputShape[2] };
            var expectedOutput = new Tensor<double>(new double[] {1.9745388 , 1.47798479, 1.88329041, 1.9745388 , 1.47798479,
                                                                  0.90630358, 1.9745388 , 1.6747036 , 2.3756249 , 1.40357327,
                                                                  1.6747036 , 2.3756249 , 0.95659649, 1.6747036 , 2.3756249 ,
                                                                  1.9745388 , 1.47798479, 2.98030591, 1.9745388 , 1.47798479,
                                                                  2.98030591, 1.9745388 , 1.30407619, 2.3756249 , 1.55503869,
                                                                  1.01333427, 2.3756249 , 0.95659649, 0.56143385, 2.3756249 ,
                                                                  2.20110965, 1.47798479, 2.98030591, 2.20110965, 1.47798479,
                                                                  2.98030591, 2.20110965, 1.30407619, 2.3756249 , 1.55503869,
                                                                  1.01333427, 2.3756249 , 1.18131149, 0.56143385, 2.3756249 ,
                                                                  2.20110965, 1.2438091 , 2.98030591, 2.20110965, 1.2438091 ,
                                                                  2.98030591, 2.20110965, 1.01333427, 1.16964471, 1.55503869,
                                                                  1.01333427, 1.49089944, 1.18131149, 0.56143385, 1.66693211,
                                                                  2.20110965, 1.2438091 , 1.57511568, 2.20110965, 1.2438091 ,
                                                                  3.1572547 , 2.20110965, 0.35566717, 3.1572547 , 2.09829068,
                                                                  0.63034421, 3.1572547 , 2.09829068, 0.63034421, 1.66693211,
                                                                  2.15504169, 0.90234876, 1.74955761, 2.15504169, 1.15027988,
                                                                  1.74955761, 1.17092967, 1.42148685, 0.05908077, 0.98024762,
                                                                  1.42148685, 0.59398502, 0.98024762, 1.42148685, 0.59398502,
                                                                  2.15504169, 0.90234876, 1.74955761, 2.15504169, 1.15027988,
                                                                  1.74955761, 1.17092967, 1.42148685, 0.05908077, 0.92222857,
                                                                  1.42148685, 0.59398502, 0.92222857, 1.42148685, 0.59398502,
                                                                  2.15504169, 0.90234876, 1.74955761, 2.15504169, 1.15027988,
                                                                  1.74955761, 0.82895947, 1.15308595, 0.20670113, 0.53881717,
                                                                  1.24700713, 0.20670113, 0.53881717, 1.24700713, 0.41602567,
                                                                  2.63162303, 0.85099638, 0.64249796, 1.64482808, 0.92508286,
                                                                  1.26315391, 1.45051908, 1.27432656, 1.26315391, 1.45051908,
                                                                  1.65289247, 1.26315391, 0.63033444, 1.65289247, 0.91875243,
                                                                  2.63162303, 0.85099638, 0.64249796, 1.45051908, 0.92508286,
                                                                  1.32079971, 1.45051908, 1.27432656, 1.32079971, 1.45051908,
                                                                  1.65289247, 1.32079971, 1.39203084, 1.65289247, 0.91875243});

            expectedOutput = expectedOutput.reshape(expectedOutputShape);

            // Act
            var poolingLayer = new PoolingLayer2D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output tensor with correct shape");
            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-6, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void PoolingLayer2D_AvgPoolingUnitStridePaddingSame_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 7, 7, 3 };
            int batchSize = 2;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1], layerInputShape[2] };
            var input = new Tensor<double>(new double[] { 0.56146491, -0.4325621 ,  0.88601032, -0.17517148,  1.38847498,
                                                         -0.08028661, -1.12155469, -0.58778088, -0.46272179, -0.52845978,
                                                          1.30353991, -1.24046951,  0.17010402,  1.67470356, -0.20598032,
                                                         -0.13742261,  0.54233763,  0.36579914, -1.52907043, -0.14506615,
                                                          0.40729955,  0.35551977,  0.86400829,  1.88329039,  0.29745863,
                                                         -0.27324774, -0.73431326,  1.3180166 , -0.17401557, -0.10312114,
                                                         -0.69645137, -1.56197892,  0.42986977, -0.60144652, -0.04465733,
                                                          0.11235045,  0.95659649,  0.11681149, -2.13546254, -0.22732446,
                                                         -0.69928343, -1.46921705, -1.44556146, -1.153496  ,  0.09489746,
                                                         -2.12854295,  1.47798476, -0.20997356,  1.97453884,  1.30407624,
                                                         -0.19798062,  1.40357332,  0.26905054,  0.90630356,  0.89581737,
                                                         -0.31213708,  2.37562488, -0.12294524,  0.46848971, -0.08462803,
                                                          0.61212434, -0.70103499,  1.17807189, -1.38645167, -0.85455524,
                                                         -0.89186198,  0.57573523, -0.96622848,  2.98030596, -2.66874266,
                                                          0.46382983,  0.72423461,  1.55503865,  1.01333431,  1.16964474,
                                                          0.95433798, -0.41045219, -1.17181812, -1.68892746,  0.56143387,
                                                          1.49089948, -0.63263902, -1.20353446,  0.10338993,  0.01262961,
                                                          0.94779568, -1.87370706,  0.06609356,  0.74285845,  0.66433993,
                                                          2.20110966, -0.57995003, -0.98022386,  0.81195219, -0.33806267,
                                                         -0.47705876, -0.04729528,  0.08869091,  0.85609904,  1.18131153,
                                                          0.32612134,  1.12162879, -0.13882688, -1.83888855,  0.05444741,
                                                          1.59519607,  0.52501207,  1.1060692 ,  0.85094447,  1.24380913,
                                                          1.57511566,  0.7152266 ,  0.35566718,  0.42239772,  1.15564331,
                                                         -0.23656388, -1.28050578, -1.65900638, -0.09968587, -0.46287323,
                                                          1.00145748, -0.56657201,  0.43335275, -0.4575495 , -0.50672402,
                                                          1.66693205,  0.18318202, -0.40510412, -2.62397045, -0.66257671,
                                                         -1.77384248,  1.50004202, -0.63067836, -0.79536606, -2.3626925 ,
                                                         -1.05587481, -0.16627958,  3.15725458, -0.65810233, -0.06300574,
                                                         -1.91743821,  2.09829067,  0.63034419, -0.0798598 , -0.52319696,
                                                          0.07392963,  0.49856004,  0.21049685, -0.08628767, -1.12611707,
                                                         -2.03317302,  0.20303325,  0.59996247, -0.5869878 ,  0.4868331 ,
                                                          0.02867377,  0.94264783,  0.69660946, -0.00520433, -0.27734673,
                                                         -0.72081539, -0.13158885,  0.9802476 ,  1.29261004,  0.50604571,
                                                          0.77498456, -0.01284429, -0.08408896, -0.60987957, -0.00892505,
                                                         -0.99751832,  0.2806547 , -1.40819646,  0.31895987,  1.17092962,
                                                         -0.91396889, -1.30201044,  0.08038122, -1.00798243, -0.10354281,
                                                         -0.25034687,  1.4214869 ,  0.05908077,  0.92222859,  0.22306133,
                                                          0.59398505, -0.29965699,  0.86203472,  0.27997115, -1.59815247,
                                                          0.90234873,  0.08397274,  2.15504169, -1.51010251,  1.74955757,
                                                          0.82895949, -2.03828956, -1.40329323, -1.02550191,  1.15027985,
                                                         -0.91570947, -1.96525968, -0.95385491, -0.75978529,  0.53881717,
                                                         -0.29598287, -0.18880006,  0.39676459, -1.50394917,  0.41602566,
                                                         -0.26422785,  0.66200554, -0.18655683,  1.64482807,  0.25808603,
                                                         -2.76706006,  0.19007231, -1.38818591, -0.07925095, -2.99019342,
                                                         -0.18778626,  0.04614819, -0.00344016,  0.39003154, -1.34984968,
                                                         -0.18631236,  1.24700716, -0.31427149,  0.45359917,  0.26096725,
                                                         -0.62056391, -1.19134459,  0.85099639,  0.64249793,  0.03684947,
                                                         -0.65883196,  0.10765061, -0.92412945,  0.60890281, -1.35575801,
                                                         -0.35015551,  0.92508287, -1.10579289, -1.17649377,  1.15308594,
                                                          0.20670113, -0.60592161, -1.06516026, -0.39640956, -0.74377697,
                                                         -0.37722864, -0.69816568,  2.63162303, -2.13000723, -1.84351966,
                                                          0.90317835, -0.1214262 ,  0.06524366, -0.02012757, -0.63654065,
                                                         -0.92797039,  1.45051906, -1.95582026,  1.26315388, -0.43424369,
                                                          1.27432657,  0.72740755,  0.63033444,  1.65289248,  0.91875244,
                                                         -1.84788191, -0.65909792, -0.32767424,  1.23565268,  0.35926131,
                                                         -0.22272   , -0.98385977, -0.75728687, -0.90239033, -0.20538704,
                                                          0.33745069,  0.33083082,  1.11875784,  0.87167551,  1.32079975,
                                                          0.97008268, -0.50337526, -1.24629487, -0.11255154,  0.07906763,
                                                         -0.54813318,  1.39203082,  0.19450113,  0.20825211});
            input = input.reshape(inputShape);

            int[] pool_size = { 3, 3 };
            int[] stride = { 1, 1 };
            PaddingType paddingType = PaddingType.same_keras;
            PoolingType poolingType = PoolingType.average;

            int[] expectedOutputShape = { batchSize, layerInputShape[0], layerInputShape[1], layerInputShape[2] };
            var expectedOutput = new Tensor<double>(new double[] { 2.59817958e-01,  3.86668354e-01,  4.88675237e-01,  2.05955625e-01,
                                                                   1.30812839e-01,  2.31476322e-01, -1.51027009e-01,  1.58319678e-02,
                                                                  -3.65173727e-01, -2.43298605e-01,  1.01635136e-01, -2.45012060e-01,
                                                                  -1.39513299e-01,  3.38459373e-01, -4.45648819e-01, -2.28093922e-01,
                                                                   2.40807652e-01, -4.87535119e-01, -2.34305233e-01, -4.63000983e-02,
                                                                  -7.07895219e-01, -4.22472090e-01,  3.11860353e-01,  3.06604147e-01,
                                                                  -4.04257514e-02,  2.68160224e-01,  1.19533472e-01,  3.81563492e-02,
                                                                   3.49567056e-01, -1.88077003e-01,  3.12682003e-01,  2.07866728e-01,
                                                                   1.79319486e-01,  1.48818418e-01,  2.72906601e-01,  5.81563897e-02,
                                                                   1.82588235e-03,  1.00018166e-01,  6.04286641e-02, -7.46736303e-02,
                                                                  -6.96242750e-02, -2.89689511e-01, -6.21973753e-01, -1.50922403e-01,
                                                                   5.20390809e-01, -3.45336616e-01,  7.64840096e-02,  3.93941998e-01,
                                                                   1.81180507e-01,  1.72533855e-01,  5.51663339e-01,  4.59409118e-01,
                                                                   6.07833080e-02,  4.71678674e-01,  2.95065969e-01,  1.10993646e-02,
                                                                   3.43642682e-01,  1.61770377e-02, -2.47151583e-01,  4.43567634e-02,
                                                                  -1.83852553e-01, -2.42852971e-01, -1.52824387e-01, -7.17683017e-01,
                                                                   3.23931985e-02,  1.27333447e-01, -3.11021358e-01,  1.53590575e-01,
                                                                   3.44478898e-02,  4.21195090e-01,  3.76321435e-01,  5.08843601e-01,
                                                                   7.86703348e-01,  1.66486636e-01,  3.56091708e-01,  5.49206972e-01,
                                                                   1.85163185e-01,  6.87410593e-01,  1.12550825e-01, -3.35701287e-01,
                                                                   6.58190489e-01, -1.31650433e-01, -3.97902161e-01,  6.43968284e-01,
                                                                   2.85691231e-01,  2.73115277e-01,  5.93376935e-01,  2.17971161e-01,
                                                                   2.08693191e-01,  4.14074481e-01,  5.84777892e-01,  1.88743755e-01,
                                                                   5.33138931e-01,  3.35362673e-01,  2.85341702e-02, -1.33344844e-01,
                                                                   3.62723559e-01,  3.75826359e-02,  1.86596543e-01, -1.65237531e-01,
                                                                  -4.05512333e-01,  4.54673141e-01, -1.22528963e-01, -5.38027287e-01,
                                                                   8.11775029e-01,  3.40911508e-01,  2.13421479e-01,  5.79815507e-02,
                                                                   4.81236398e-01,  2.89866254e-02, -2.85847723e-01,  3.83537769e-01,
                                                                  -1.71969980e-01,  2.46518776e-01,  9.25527364e-02, -2.03839511e-01,
                                                                  -3.38337868e-01,  3.14264029e-01, -4.72236946e-02,  1.50066599e-01,
                                                                   8.85647014e-02, -2.17310011e-01,  2.41205424e-01,  5.26914358e-01,
                                                                  -3.13631594e-01,  6.15843534e-01,  4.91686463e-01, -1.02531344e-01,
                                                                   3.89314055e-01,  3.41882348e-01, -1.41637385e-01, -6.38397932e-02,
                                                                   6.21140786e-02, -2.28762612e-01,  5.01935303e-01, -3.55465323e-01,
                                                                  -1.67538986e-01, -4.07309562e-01,  1.47067979e-01, -8.36271346e-02,
                                                                  -2.50116084e-02, -3.30178328e-02, -8.86189565e-02,  2.31122617e-02,
                                                                   5.29750407e-01, -9.22555551e-02,  6.29746258e-01, -5.37975252e-01,
                                                                  -3.25093985e-01, -3.01178277e-01, -2.61326522e-01, -2.87918597e-01,
                                                                  -4.13008302e-01, -2.42579076e-02, -3.23945314e-01, -7.71935806e-02,
                                                                   1.79879561e-01, -6.30619144e-03, -2.42431983e-01,  3.99635285e-01,
                                                                   3.17494988e-01,  1.53129250e-01,  3.08351696e-01,  5.10922253e-01,
                                                                   2.03900814e-01,  5.94450951e-01,  5.91215432e-01,  3.23978215e-01,
                                                                  -2.65835285e-01, -3.18021625e-01,  1.04802869e-01, -2.02344786e-02,
                                                                  -4.85950589e-01, -2.27534741e-01,  2.01439083e-01, -4.82420444e-01,
                                                                  -1.14734069e-01, -1.20280534e-01, -2.08855733e-01, -5.03708899e-01,
                                                                  -6.01475779e-03,  2.00601310e-01, -1.05057701e-01,  9.11591351e-02,
                                                                   3.46385092e-02,  7.67605752e-02,  5.52230895e-01,  9.41549540e-02,
                                                                   2.53856391e-01,  2.68044084e-01, -1.84130594e-01, -2.99774170e-01,
                                                                   4.22025084e-01, -6.05025351e-01, -5.09244382e-01,  2.59463578e-01,
                                                                  -7.82905042e-01, -4.95133460e-01, -4.40488815e-01, -3.92029971e-01,
                                                                  -6.45356953e-01, -5.42180777e-01,  2.20695585e-01, -3.25860560e-01,
                                                                  -4.37340625e-02,  1.83422446e-01, -2.09356442e-01,  3.04240048e-01,
                                                                   1.32189751e-01,  2.77243946e-02,  1.30499050e-01,  8.40836987e-02,
                                                                  -6.16563298e-02,  9.75440815e-02, -2.57007807e-01, -3.56471121e-01,
                                                                  -4.82476912e-02, -3.15649390e-01, -6.35945380e-01, -8.24015856e-01,
                                                                  -3.78592797e-02, -7.46287763e-01, -8.62717867e-01,  2.62522578e-01,
                                                                  -5.30863225e-01, -3.65780383e-01, -1.27231553e-01, -4.11679894e-01,
                                                                  -2.44716704e-02, -2.89057761e-01, -3.00364166e-01,  6.26817703e-01,
                                                                  -1.89862922e-01, -6.63624048e-01,  3.34080189e-01, -2.83889025e-01,
                                                                  -7.04969287e-01, -6.57318719e-03, -3.50724369e-01, -5.28181791e-01,
                                                                  -4.73132461e-01,  2.03440860e-02, -2.86134601e-01, -4.07323033e-01,
                                                                   3.81517768e-01, -4.62267140e-04, -4.34904099e-01,  4.30758238e-01,
                                                                  -2.06008181e-01, -3.83326530e-01,  1.76563337e-01, -2.39722073e-01,
                                                                   4.38683182e-01, -4.09549087e-01, -3.58872980e-01,  1.64717242e-01,
                                                                  -2.38609076e-01, -4.56237257e-01,  1.13960609e-01, -1.54088199e-01,
                                                                  -1.33803651e-01,  4.76469658e-02,  2.30532005e-01, -8.74358788e-02,
                                                                   1.65592015e-01,  2.70197272e-01,  1.26687124e-01, -2.14269072e-01,
                                                                   1.94334626e-01, -1.28396049e-01, -2.14627758e-01, -2.91709248e-02,
                                                                  -1.40563026e-01,  9.46648657e-01, -6.62364721e-01, -7.25846529e-01,
                                                                   5.93513310e-01, -4.91424799e-01, -5.83420932e-01,  3.77180129e-01,
                                                                  -3.76991272e-01,  1.91611230e-01,  4.79933590e-01, -1.02047242e-01,
                                                                   2.44654432e-01,  6.03816450e-01,  2.36461103e-01,  4.05947566e-01,
                                                                   9.96284857e-02,  3.39719057e-01, -4.46150303e-02,  1.54829621e-02,
                                                                   3.16840827e-01,  6.27992749e-02});

            expectedOutput = expectedOutput.reshape(expectedOutputShape);

            // Act
            var poolingLayer = new PoolingLayer2D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output tensor with correct shape");
            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-7, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void PoolingLayer2D_AvgPoolingNonUnitStridePaddingSame_CheckResultShapeAndValues()
        {
            // Arrange
            int[] layerInputShape = { 7, 7, 3 };
            int batchSize = 2;
            int[] inputShape = { batchSize, layerInputShape[0], layerInputShape[1], layerInputShape[2] };
            var input = new Tensor<double>(new double[] { 0.56146491, -0.4325621 ,  0.88601032, -0.17517148,  1.38847498,
                                                         -0.08028661, -1.12155469, -0.58778088, -0.46272179, -0.52845978,
                                                          1.30353991, -1.24046951,  0.17010402,  1.67470356, -0.20598032,
                                                         -0.13742261,  0.54233763,  0.36579914, -1.52907043, -0.14506615,
                                                          0.40729955,  0.35551977,  0.86400829,  1.88329039,  0.29745863,
                                                         -0.27324774, -0.73431326,  1.3180166 , -0.17401557, -0.10312114,
                                                         -0.69645137, -1.56197892,  0.42986977, -0.60144652, -0.04465733,
                                                          0.11235045,  0.95659649,  0.11681149, -2.13546254, -0.22732446,
                                                         -0.69928343, -1.46921705, -1.44556146, -1.153496  ,  0.09489746,
                                                         -2.12854295,  1.47798476, -0.20997356,  1.97453884,  1.30407624,
                                                         -0.19798062,  1.40357332,  0.26905054,  0.90630356,  0.89581737,
                                                         -0.31213708,  2.37562488, -0.12294524,  0.46848971, -0.08462803,
                                                          0.61212434, -0.70103499,  1.17807189, -1.38645167, -0.85455524,
                                                         -0.89186198,  0.57573523, -0.96622848,  2.98030596, -2.66874266,
                                                          0.46382983,  0.72423461,  1.55503865,  1.01333431,  1.16964474,
                                                          0.95433798, -0.41045219, -1.17181812, -1.68892746,  0.56143387,
                                                          1.49089948, -0.63263902, -1.20353446,  0.10338993,  0.01262961,
                                                          0.94779568, -1.87370706,  0.06609356,  0.74285845,  0.66433993,
                                                          2.20110966, -0.57995003, -0.98022386,  0.81195219, -0.33806267,
                                                         -0.47705876, -0.04729528,  0.08869091,  0.85609904,  1.18131153,
                                                          0.32612134,  1.12162879, -0.13882688, -1.83888855,  0.05444741,
                                                          1.59519607,  0.52501207,  1.1060692 ,  0.85094447,  1.24380913,
                                                          1.57511566,  0.7152266 ,  0.35566718,  0.42239772,  1.15564331,
                                                         -0.23656388, -1.28050578, -1.65900638, -0.09968587, -0.46287323,
                                                          1.00145748, -0.56657201,  0.43335275, -0.4575495 , -0.50672402,
                                                          1.66693205,  0.18318202, -0.40510412, -2.62397045, -0.66257671,
                                                         -1.77384248,  1.50004202, -0.63067836, -0.79536606, -2.3626925 ,
                                                         -1.05587481, -0.16627958,  3.15725458, -0.65810233, -0.06300574,
                                                         -1.91743821,  2.09829067,  0.63034419, -0.0798598 , -0.52319696,
                                                          0.07392963,  0.49856004,  0.21049685, -0.08628767, -1.12611707,
                                                         -2.03317302,  0.20303325,  0.59996247, -0.5869878 ,  0.4868331 ,
                                                          0.02867377,  0.94264783,  0.69660946, -0.00520433, -0.27734673,
                                                         -0.72081539, -0.13158885,  0.9802476 ,  1.29261004,  0.50604571,
                                                          0.77498456, -0.01284429, -0.08408896, -0.60987957, -0.00892505,
                                                         -0.99751832,  0.2806547 , -1.40819646,  0.31895987,  1.17092962,
                                                         -0.91396889, -1.30201044,  0.08038122, -1.00798243, -0.10354281,
                                                         -0.25034687,  1.4214869 ,  0.05908077,  0.92222859,  0.22306133,
                                                          0.59398505, -0.29965699,  0.86203472,  0.27997115, -1.59815247,
                                                          0.90234873,  0.08397274,  2.15504169, -1.51010251,  1.74955757,
                                                          0.82895949, -2.03828956, -1.40329323, -1.02550191,  1.15027985,
                                                         -0.91570947, -1.96525968, -0.95385491, -0.75978529,  0.53881717,
                                                         -0.29598287, -0.18880006,  0.39676459, -1.50394917,  0.41602566,
                                                         -0.26422785,  0.66200554, -0.18655683,  1.64482807,  0.25808603,
                                                         -2.76706006,  0.19007231, -1.38818591, -0.07925095, -2.99019342,
                                                         -0.18778626,  0.04614819, -0.00344016,  0.39003154, -1.34984968,
                                                         -0.18631236,  1.24700716, -0.31427149,  0.45359917,  0.26096725,
                                                         -0.62056391, -1.19134459,  0.85099639,  0.64249793,  0.03684947,
                                                         -0.65883196,  0.10765061, -0.92412945,  0.60890281, -1.35575801,
                                                         -0.35015551,  0.92508287, -1.10579289, -1.17649377,  1.15308594,
                                                          0.20670113, -0.60592161, -1.06516026, -0.39640956, -0.74377697,
                                                         -0.37722864, -0.69816568,  2.63162303, -2.13000723, -1.84351966,
                                                          0.90317835, -0.1214262 ,  0.06524366, -0.02012757, -0.63654065,
                                                         -0.92797039,  1.45051906, -1.95582026,  1.26315388, -0.43424369,
                                                          1.27432657,  0.72740755,  0.63033444,  1.65289248,  0.91875244,
                                                         -1.84788191, -0.65909792, -0.32767424,  1.23565268,  0.35926131,
                                                         -0.22272   , -0.98385977, -0.75728687, -0.90239033, -0.20538704,
                                                          0.33745069,  0.33083082,  1.11875784,  0.87167551,  1.32079975,
                                                          0.97008268, -0.50337526, -1.24629487, -0.11255154,  0.07906763,
                                                         -0.54813318,  1.39203082,  0.19450113,  0.20825211});
            input = input.reshape(inputShape);

            int[] pool_size = { 3, 3 };
            int[] stride = { 2, 3 };
            PaddingType paddingType = PaddingType.same_keras;
            PoolingType poolingType = PoolingType.average;

            int[] expectedOutputShape = { batchSize, (int)Math.Ceiling((double)layerInputShape[0]/stride[0]), 
                (int)Math.Ceiling((double)layerInputShape[1] / stride[1]), layerInputShape[2] };
            var expectedOutput = new Tensor<double>(new double[] { 0.25981796,  0.38666835,  0.48867524, -0.2432986 ,  0.10163514,
                                                                  -0.24501206, -0.23430523, -0.0463001 , -0.7078952 , -0.62197375,
                                                                  -0.1509224 ,  0.5203908 ,  0.45940912,  0.06078331,  0.47167867,
                                                                  -0.18385255, -0.24285297, -0.15282439,  0.28569123,  0.27311528,
                                                                   0.59337693,  0.33536267,  0.02853417, -0.13334484, -0.12252896,
                                                                  -0.5380273 ,  0.811775  ,  0.49168646, -0.10253134,  0.38931406,
                                                                  -0.35546532, -0.16753899, -0.40730956,  0.5297504 , -0.09225556,
                                                                   0.62974626, -0.53797525, -0.32509398, -0.30117828,  0.17987956,
                                                                  -0.00630619, -0.24243198,  0.59445095,  0.59121543,  0.32397822,
                                                                   0.26804408, -0.1841306 , -0.29977417, -0.44048882, -0.39202997,
                                                                  -0.64535695,  0.30424005,  0.13218975,  0.02772439,  0.6268177 ,
                                                                  -0.18986292, -0.66362405, -0.47313246,  0.02034409, -0.2861346 ,
                                                                  -0.38332653,  0.17656334, -0.23972207,  0.94664866, -0.6623647 ,
                                                                  -0.7258465 ,  0.4799336 , -0.10204724,  0.24465443,  0.01548296,
                                                                   0.31684083,  0.06279927});

            expectedOutput = expectedOutput.reshape(expectedOutputShape);

            // Act
            var poolingLayer = new PoolingLayer2D<double>(layerInputShape, poolingType, pool_size, stride, paddingType);
            var output = poolingLayer.FeedForward(input);
            var actualShape = output.Shape;

            // Assert
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output tensor with correct shape");
            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < tmp0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-7, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void Flatten_0D_CheckResultShapeAndValue()
        {
            // Arrange
            int batchSize = 128;
            var input = Tensor<double>.randNormal(0, 1, batchSize);
            var expectedOutput = input.reshape(batchSize, 1);
            int[] expectedOutputShape = expectedOutput.Shape;

            // Act
            var flatten = new Flatten<double>(new int[0]);
            var output = flatten.FeedForward(input);
            var actualShape = output.Shape;

            // Assert 
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output tensor with correct shape");
            for (int b = 0; b < batchSize; b++)
                Assert.AreEqual(expectedOutput[b, 0], output[b, 0], 1e-7, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void Flatten_1D_CheckResultShapeAndValue()
        {
            // Arrange
            int batchSize = 128;
            int[] layerInputShape = { 100 };
            var input = Tensor<double>.randNormal(0, 1, batchSize, layerInputShape[0]);
            var expectedOutput = input;
            int[] expectedOutputShape = expectedOutput.Shape;

            // Act
            var flatten = new Flatten<double>(layerInputShape);
            var output = flatten.FeedForward(input);
            var actualShape = output.Shape;

            // Assert 
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output tensor with correct shape");
            for (int b = 0; b < expectedOutputShape[0]; b++)
                for (int i = 0; i < expectedOutputShape[1]; i++)
                    Assert.AreEqual(expectedOutput[b, i], output[b, i], 1e-7, "Failed to produce output tensor with correct values");
        }

        [Test]
        public void Flatten_ND_CheckResultShapeAndValue()
        {
            // Arrange
            int batchSize = 128;
            int[] layerInputShape = { 100, 100, 3 };
            var input = Tensor<double>.randNormal(0, 1, batchSize, layerInputShape[0], layerInputShape[1], layerInputShape[2]);
            var expectedOutput = input.reshape(batchSize, layerInputShape.Aggregate((a, b) => a * b));
            int[] expectedOutputShape = expectedOutput.Shape;

            // Act
            var flatten = new Flatten<double>(layerInputShape);
            var output = flatten.FeedForward(input);
            var actualShape = output.Shape;

            // Assert 
            CollectionAssert.AreEqual(expectedOutputShape, actualShape, "Failed to produce output tensor with correct shape");
            for (int b = 0; b < expectedOutputShape[0]; b++)
                for (int i = 0; i < expectedOutputShape[1]; i++)
                            Assert.AreEqual(expectedOutput[b, i], output[b, i], 1e-7, "Failed to produce output tensor with correct values");
        }
    }
}
