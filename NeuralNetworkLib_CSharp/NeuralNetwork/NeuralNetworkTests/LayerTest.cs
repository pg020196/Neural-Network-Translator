using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensor;
using Layers;


namespace NeuralNetworkTests
{
    [TestClass]
    public class TestLayer
    {
        [TestMethod]
        public void InputLayer_CheckResultShapeAndValues()
        {
            // Arrage
            int[] inputShape = { 100, 100, 3 };
            int batchSize = 32;
            int[] batchShape = { 32, 100, 100, 3 };
            var input = Tensor<double>.randUniform(0, 1, batchShape);

            // Act
            var inputLayer = new InputLayer<double>(inputShape);

            // Assert
            var output = inputLayer.FeedForward(input);
            CollectionAssert.AreEqual(batchShape, output.Shape, "Failed to produce output tensor of correct shape");
            var tmp0 = input.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < input.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output tensor with correct values");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void InputLayer_0DInput_ShouldThrowArgumentException()
        {
            // Arrage
            int[] inputShape = { 100, 100, 3 };
            int[] batchShape = { 10 }; // batchShape should have 1 dimension more than inputShape --> Exception
            var input = Tensor<double>.randUniform(0, 1, batchShape);

            // Act
            var inputLayer = new InputLayer<double>(inputShape);

            // Assert
            inputLayer.FeedForward(input);
        }


        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void InputLayer_WrongShape_ShouldThrowArgumentException()
        {
            // Arrage
            int[] inputShape = { 100, 100, 3 };
            int[] batchShape = inputShape; // batchShape should have 1 dimension more than inputShape --> Exception
            var input = Tensor<double>.randUniform(0, 1, batchShape);

            // Act
            var inputLayer = new InputLayer<double>(inputShape);

            // Assert
            inputLayer.FeedForward(input);
        }

        [TestMethod]
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

            // Assert
            var output = dense.FeedForward(input);
            var actualOutputShape = output.Shape;
            CollectionAssert.AreEqual(expectedOuputShape, actualOutputShape, "Failed to produce output tensor with correct shape");
            var tmp0 = expectedOutput.flatten();
            var tmp1 = output.flatten();
            for (int i = 0; i < expectedOutput.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce output tensor with correct values");
        }
    }
}
