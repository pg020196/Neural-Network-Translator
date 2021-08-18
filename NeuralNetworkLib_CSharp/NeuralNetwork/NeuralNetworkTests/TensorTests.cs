using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensor;

namespace NeuralNetworkTests
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod]
        public void Constructor_CreateFrom0DInitializer_CheckFields()
        {
            // Arrange
            int[] expectedShape = new int[0];
            int expectedDims = 0;
            int expectedNumElems = 1;
            double initValue = 5.7d;

            // Act
            var t0 = new Tensor<double>(initValue);

            // Assert
            int[] actualShape = t0.Shape;
            int actualDims = t0.Dims;
            int actualNumElems = t0.NumElems;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Shape not derived correctly");
            Assert.AreEqual(expectedDims, actualDims, "Number of dimensions not derived correctly");
            Assert.AreEqual(expectedNumElems, actualNumElems, "Number of elements not derived correctly");
        }


        [TestMethod]
        public void Constructor_CreateFrom1DInitializer_CheckFields()
        {
            // Arrange
            float[] initializer1D = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
            int[] expectedShape = new int[1] { 6 };
            int expectedDims = 1;
            int expectedNumElems = 6;
            float expectedValue = 5f;

            // Act
            var t0 = new Tensor<float>(initializer1D);

            // Assert
            int[] actualShape = t0.Shape;
            int actualDims = t0.Dims;
            int actualNumElems = t0.NumElems;
            float actualValue = t0[4];
            CollectionAssert.AreEqual(expectedShape, actualShape, "Shape not derived correctly");
            Assert.AreEqual(expectedDims, actualDims, "Number of dimensions not derived correctly");
            Assert.AreEqual(expectedNumElems, actualNumElems, "Number of elements not derived correctly");
            Assert.AreEqual(expectedValue, actualValue, "Did not correctly initialize data");
        }


        [TestMethod]
        public void Constructor_CreateFromOtherTensorWithCopyAndModify_ShouldNotModifyOriginal()
        {
            // Arrange
            double[] initializer1D = new double[] { 10d, 10d, 10d, 10d };
            Tensor<double> t0 = new Tensor<double>(initializer1D);
            double expectedValue = 10d;
            var t1 = new Tensor<double>(t0, copy: true);

            // Act
            t1[0] = -1d;

            // Assert
            double actualValue = t0[0];
            Assert.AreEqual(expectedValue, actualValue, 1e-8, "New instance not using copy of data");
        }

        [TestMethod]
        public void Constructor_CreateFromOtherTensorWithoutCopyAndModify_ShouldModifyOriginal()
        {
            // Arrange
            double[] initializer1D = new double[] { 10d, 10d, 10d, 10d };
            Tensor<double> t0 = new Tensor<double>(initializer1D);
            var t1 = new Tensor<double>(t0, copy: false);

            // Act
            t1[0] = -1d;
            double expectedValue = -1d;

            // Assert
            double actualValue = t0[0];
            Assert.AreEqual(expectedValue, actualValue, 1e-8, "New instance not referencing original data");
        }

        //[TestMethod]
        //public void GetSlice_With


        [TestMethod]
        public void GetSlice_WithValidAxisAndSingletonRange_CheckResultShape()
        {
            // Assert
            int[] shape = { 3, 4, 5, 6 };
            var t0 = Tensor<double>.randNormal(0, 1, shape);
            int axis = 0;
            Range range = 0..1;
            int[] expectedShape_keepDims = new int[] { 1, 4, 5, 6 };
            int[] expectedShape_dropDims = new int[] { 4, 5, 6 };

            // Act 
            var t0_keepDims = t0.getSlice(axis, range, keepdims: true);
            var t0_dropDims = t0.getSlice(axis, range, keepdims: false);

            // Assert
            int[] actualShape_keepDims = t0_keepDims.Shape;
            int[] actualShape_dropDims = t0_dropDims.Shape;
            CollectionAssert.AreEqual(expectedShape_keepDims, actualShape_keepDims, "Failed to produce correct shape of slice " +
                "when keeping singleton dimension");
            CollectionAssert.AreEqual(expectedShape_dropDims, actualShape_dropDims, "Failed to produce correct shape of slice " +
                "when dropping singleton dimension");

            for (int i = 0; i < shape[1]; i++)
                for (int j = 0; j < shape[2]; j++)
                    for (int k = 0; k < shape[3]; k++)
                    {
                        Assert.AreEqual(t0_keepDims[0, i, j, k], t0[0, i, j, k], 1e-5, "Failed to address correct elements in slice");
                        Assert.AreEqual(t0_dropDims[i, j, k], t0[0, i, j, k], 1e-5, "Failed to address correct elements in slice");
                    }
        }

        [TestMethod]
        public void GetSlice_WithValidAxisAndNonSingletonRange_CheckResultsShapeAndValues()
        {
            // Arrange
            int[] shape = { 3, 4, 6 };
            var t0 = Tensor<double>.randUniform(0, 1, shape);
            int axis = 2;
            Range range = 1..5; // i.e. {1, 2, 3, 4,}, end is exclusive
            int[] expectedShape = new int[] { 3, 4, 4 };

            // Act
            var t0_slice = t0.getSlice(axis, range);

            // Assert
            int[] actualShape = t0_slice.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to produce correct shape of slice ");
            for (int i = 0; i < shape[0]; i++)
                for (int j = 0; j < shape[1]; j++)
                    for (int k = 0; k < (range.End.Value - range.Start.Value); k++)
                        Assert.AreEqual(t0_slice[i, j, k], t0[i, j, k + range.Start.Value], 1e-5, "Failed to address correct elements in slice");
        }

        [TestMethod]
        public void GetSlice_WithNegativeAxis_CheckAxisHandling()
        {
            // Arrange
            int[] shape = { 4, 6, 9, 3 };
            var t0 = Tensor<double>.randNormal(0, 1, shape);
            int axisFromEnd = -3;
            int axisFromStart = 1;
            Range range = 2..4;
            int[] expectedShape = { 4, 2, 9, 3 };

            // Act 
            var t0_slice_axisFromEnd = t0.getSlice(axisFromEnd, range);
            var t0_slice_axisFromStart = t0.getSlice(axisFromStart, range);

            // Assert
            int[] actualShape_FromEnd = t0_slice_axisFromEnd.Shape;
            int[] actualShape_FromStart = t0_slice_axisFromStart.Shape;
            CollectionAssert.AreEqual(actualShape_FromStart, actualShape_FromEnd, "Failed to produce correct shape of slice");
            CollectionAssert.AreEqual(expectedShape, actualShape_FromEnd, "Failed to produce correct shape of slice");
            for (int i = 0; i < shape[0]; i++)
                for (int j = 0; j < range.End.Value - range.Start.Value; j++)
                    for (int k = 0; k < shape[2]; k++)
                        for (int l = 0; l < shape[3]; l++)
                            Assert.AreEqual(t0_slice_axisFromEnd[i, j, k, l], t0[i, j + range.Start.Value, k, l], 1e-8,
                                "Failed to address correct elements in slice");
        }

        [TestMethod]
        public void GetSlice_WithRangeFromEnd_CheckRangeHandling()
        {
            // Arrange
            int[] shape = { 34, 28, 3 };
            var t0 = Tensor<float>.zeros(shape);
            int axis = 1;
            Range range_fromEnd = ^5..^1;
            Range range_fromStart = 23..27;
            int[] expectedShape = { 34, 4, 3 };

            // Act
            var t0_slice_fromEnd = t0.getSlice(axis, range_fromEnd);
            var t0_slice_fromStart = t0.getSlice(axis, range_fromStart);

            // Assert
            int[] actualShape_FromEnd = t0_slice_fromEnd.Shape;
            int[] actualShape_FromStart = t0_slice_fromStart.Shape;
            CollectionAssert.AreEqual(actualShape_FromEnd, actualShape_FromStart, "Failed to produce correct shape of slice");
            CollectionAssert.AreEqual(expectedShape, actualShape_FromEnd, "Failed to produce correct shape of slice");
            for (int i = 0; i < shape[0]; i++)
                for (int j = 0; j < range_fromStart.End.Value - range_fromStart.Start.Value; j++)
                    for (int k = 0; k < shape[2]; k++)
                        Assert.AreEqual(t0_slice_fromEnd[i, j, k], t0[i, j + range_fromStart.Start.Value, k], 1e-5,
                            "Failed to address correct elements in slice");
        }


        [TestMethod]
        public void GetSlice_ModifyElementInSlice_ShouldModifyOriginalAsWell()
        {
            // Arrange
            int[] shape = { 4, 4, 8 };
            var t0 = Tensor<double>.ones(shape);
            int axis = 2;
            Range range = 2..5;
            var t1 = t0.getSlice(axis, range);

            // Act
            t1[0, 0, 0] = -8d;
            double expectedValue = t1[0, 0, 0];

            // Assert
            double actualValue = t0[0, 0, range.Start.Value];
            Assert.AreEqual(expectedValue, actualValue, 1e-8, "Failed to reference underlying data");

        }


        [TestMethod]
        public void Min_AllAxes_CheckResult()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] { 1.5, 2.7, 2.5, 8.9, 17.0, 23.0, 7.9, 19.2, 26.1, -3.5, -11.8, 12.0 });
            t0 = t0.reshape(shape);
            var expectedValue = -11.8;
            var expectedShape = new int[0];

            // Act
            var min = t0.min();

            // Assert
            int[] actualShape = min.Shape;
            double actualValue = min[0];
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing min");
            Assert.AreEqual(expectedValue, actualValue, 1e-8, "Failed to determine correct min value");
        }

        [TestMethod]
        public void Min_OneAxis_CheckResult()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] { 1.5,  2.7,  2.5,   8.9,
                                                      17.0, 23.0,  7.9,  19.2,
                                                      26.1, -3.5, -11.8, 12.0});
            t0 = t0.reshape(shape);
            int axis = -1;
            var expectedValue = new Tensor<double>(new double[] { 1.5, 7.9, -11.8 });
            int[] expectedShape = { shape[0] };

            // Act
            var min = t0.min(axis);

            // Assert
            int[] actualShape = min.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing min");
            for (int i = 0; i < shape[0]; i++)
                Assert.AreEqual(expectedValue[i], min[i], 1e-8, "Failed to determine correct min value");
        }

        [TestMethod]
        public void Min_MultipleAxes_CheckResult()
        {
            // Arrange
            int[] shape = { 2, 2, 3 };
            var t0 = new Tensor<double>(new double[] { 1.5,  2.7,   2.5,
                                                       8.9,  17.0, 23.0,
                                                       7.9,  19.2, 26.1,
                                                      -3.5, -11.8, 12.0});
            t0 = t0.reshape(shape);
            int[] axes = { 1, 2 };
            var expectedValue = new Tensor<double>(new double[] { 1.5, -11.8 });
            int[] expectedShape = { shape[0] };

            // Act
            var min = t0.min(axes);

            // Assert
            int[] actualShape = min.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing min");
            for (int i = 0; i < shape[0]; i++)
                Assert.AreEqual(expectedValue[i], min[i], 1e-8, "Failed to determine correct min value");
        }


        [TestMethod]
        public void Mean_AllAxes_CheckResult()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.61661296, 1.40819835, -1.02500508,   1.04534558, -1.78292752,
                                                       0.6544806,  0.15202791,  0.90817283,   1.89842621, -0.04575441,
                                                      -0.18184348, -0.61511024, -0.05284911, -0.08826166, -2.24415534});
            t0 = t0.reshape(shape);
            double expectedValue = -0.03905788862424567;
            var expectedShape = new int[0];

            // Act
            var mean = t0.mean();

            // Assert
            int[] actualShape = mean.Shape;
            double actualValue = mean[0];
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing mean");
            Assert.AreEqual(expectedValue, actualValue, 1e-8, "Failed to determine correct mean value");
        }

        [TestMethod]
        public void Mean_OneAxis_CheckResult()
        {
            // Arrange
            int[] shape = { 2, 5, 2 };
            var t0 = new Tensor<double>(new double[] {0.21450223,  1.43700387,  0.78428915, -3.20674129, -0.59453971,
                                                      0.9791911,  -0.54945224, -1.23064184,  1.32190338,  0.30240403,
                                                     -0.8653634,   0.22764933, -0.77571159,  0.64463162,  0.07112009,
                                                     -0.61411165, -0.38939276,  1.12518949,  1.52588014, -0.08618781 });
            t0 = t0.reshape(shape);
            int axis = 1;
            int[] expectedShape = { 2, 2 };
            var expectedValue = new Tensor<double>(new double[] { 0.23534056, -0.34375683, -0.0866935, 0.2594342 }).reshape(expectedShape);


            // Act 
            var mean = t0.mean(axis);

            // Assert
            int[] actualShape = mean.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing mean");
            for (int i = 0; i < shape[0]; i++)
                for (int j = 0; j < shape[2]; j++)
                    Assert.AreEqual(expectedValue[i, j], mean[i, j], 1e-8, "Failed to determine correct mean value");
        }

        [TestMethod]
        public void Mean_MultipleAxes_CheckResult()
        {
            // Arrange
            int[] shape = { 2, 5, 2 };
            var t0 = new Tensor<double>(new double[] {0.21450223,  1.43700387,  0.78428915, -3.20674129, -0.59453971,
                                                      0.9791911,  -0.54945224, -1.23064184,  1.32190338,  0.30240403,
                                                     -0.8653634,   0.22764933, -0.77571159,  0.64463162,  0.07112009,
                                                     -0.61411165, -0.38939276,  1.12518949,  1.52588014, -0.08618781 });
            t0 = t0.reshape(shape);
            int[] axis = { 1, 2 };
            int[] expectedShape = { 2 };
            var expectedValue = new Tensor<double>(new double[] { -0.05420813, 0.08637035 });

            // Act 
            var mean = t0.mean(axis);

            // Assert
            int[] actualShape = mean.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing mean");
            for (int i = 0; i < shape[0]; i++)
                Assert.AreEqual(expectedValue[i], mean[i], 1e-8, "Failed to determine correct mean value");
        }

        [TestMethod]
        public void Max_AllAxes_CheckResult()
        {
            // Arrange
            int[] shape = { 2, 4, 5 };
            var t0 = new Tensor<double>(new double[] {0.51664251, -2.55989291,  1.73084967,  1.72702079,  0.08573073,
                                                     -1.5121638 , -1.12582488, -0.68386640, -0.35690710, -0.44542221,
                                                     -0.6511695 ,  0.60908527,  1.56157053,  1.49976867, -0.55472341,
                                                     -0.00306336,  0.68158875, -0.69587351, -0.40892407, -1.96886228,
                                                      2.15764942, -0.44870526, -0.35155568, -0.76801771, -1.88551986,
                                                      0.86981032, -0.02458815, -0.59420657, -1.12125096, -0.57282594,
                                                     -0.13347450,  1.97409950, -0.58730783,  0.15157498,  0.97848569,
                                                     -1.91258565,  0.06975641, -0.84319698, -0.19206467, -0.70831456});
            t0 = t0.reshape(shape);
            double expectedValue = 2.15764942;
            var expectedShape = new int[0];

            // Act
            var max = t0.max();

            // Assert
            int[] actualShape = max.Shape;
            double actualValue = max[0];
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing max");
            Assert.AreEqual(expectedValue, actualValue, 1e-8, "Failed to determine correct max value");
        }

        [TestMethod]
        public void Max_OneAxis_CheckResult()
        {
            // Arrange
            int[] shape = { 2, 4, 5 };
            var t0 = new Tensor<double>(new double[] {0.51664251, -2.55989291,  1.73084967,  1.72702079,  0.08573073,
                                                     -1.51216380, -1.12582488, -0.68386640, -0.35690710, -0.44542221,
                                                     -0.65116950,  0.60908527,  1.56157053,  1.49976867, -0.55472341,
                                                     -0.00306336,  0.68158875, -0.69587351, -0.40892407, -1.96886228,
                                                      2.15764942, -0.44870526, -0.35155568, -0.76801771, -1.88551986,
                                                      0.86981032, -0.02458815, -0.59420657, -1.12125096, -0.57282594,
                                                     -0.13347450,  1.97409950, -0.58730783,  0.15157498,  0.97848569,
                                                     -1.91258565,  0.06975641, -0.84319698, -0.19206467, -0.70831456});
            t0 = t0.reshape(shape);
            int axis = 1;
            int[] expectedShape = { 2, 5 };
            var expectedValue = new Tensor<double>(new double[] {0.51664251,  0.68158875,  1.73084967,  1.72702079,  0.08573073,
                                                                 2.15764942,  1.9740995 , -0.35155568,  0.15157498,  0.97848569}).reshape(expectedShape);

            // Act
            var max = t0.max(axis);

            // Assert
            int[] actualShape = max.Shape;
            //double actualValue = max[0];
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing max");
            for (int i=0; i<t0.Shape[0]; i++)
                for (int j=0; j<t0.Shape[1]; j++)
                    Assert.AreEqual(expectedValue[i,j], max[i,j], 1e-8, "Failed to determine correct max value");
        }

        [TestMethod]
        public void Max_MultipleAxis_CheckResult()
        {
            // Arrange
            int[] shape = { 2, 4, 5 };
            var t0 = new Tensor<double>(new double[] {0.51664251, -2.55989291,  1.73084967,  1.72702079,  0.08573073,
                                                     -1.51216380, -1.12582488, -0.68386640, -0.35690710, -0.44542221,
                                                     -0.65116950,  0.60908527,  1.56157053,  1.49976867, -0.55472341,
                                                     -0.00306336,  0.68158875, -0.69587351, -0.40892407, -1.96886228,
                                                      2.15764942, -0.44870526, -0.35155568, -0.76801771, -1.88551986,
                                                      0.86981032, -0.02458815, -0.59420657, -1.12125096, -0.57282594,
                                                     -0.13347450,  1.97409950, -0.58730783,  0.15157498,  0.97848569,
                                                     -1.91258565,  0.06975641, -0.84319698, -0.19206467, -0.70831456});
            t0 = t0.reshape(shape);
            int[] axes = { 0, 2 };
            int[] expectedShape = { 4 };
            var expectedResult = new Tensor<double>(new double[] { 2.15764942, 0.86981032, 1.9740995, 0.68158875 });

            // Act
            var max = t0.max(axes);

            // Assert
            int[] actualShape = max.Shape;
            //double actualValue = max[0];
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing max");
            for (int i = 0; i < t0.Shape[1]; i++)
                    Assert.AreEqual(expectedResult[i], max[i], 1e-8, "Failed to determine correct max value");
        }


        [TestMethod]
        public void Sum_AllAxes_CheckResult()
        {
            // Arrange
            int[] shape = { 3, 6, 2 };
            var t0 = new Tensor<double>(new double[] {-0.06314949,  0.19945749,  0.44094216,  0.52029016,  1.55916992,
                                                       0.43801613, -0.83891018,  0.72300822,  0.42764385,  0.48444319,
                                                       0.34606469, -0.31548898,  0.34424614,  0.07873913, -0.91834060,
                                                      -0.17023571,  0.15497373, -1.42028829, -0.50732354,  0.38278548,
                                                      -0.63573374, -0.53303308,  0.27223804, -0.20991905,  0.45978979,
                                                      -0.60859952,  0.78597102, -0.31466016, -0.50330494,  0.07287572,
                                                       1.30384166, -0.15242888,  0.32414952,  0.99723693, -1.27655555,
                                                      -1.21717734});
            t0 = t0.reshape(shape);
            double expectedValue = 0.6307339200000002;
            var expectedShape = new int[0];

            // Act
            var sum = t0.sum();

            // Assert
            int[] actualShape = sum.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing sum");
            Assert.AreEqual(expectedValue, sum[0], 1e-8, "Failed to determine correct sum value");
        }

        [TestMethod]
        public void Sum_OneAxis_CheckResult()
        {
            // Arrange
            int[] shape = { 3, 6, 2 };
            var t0 = new Tensor<double>(new double[] {-0.06314949,  0.19945749,  0.44094216,  0.52029016,  1.55916992,
                                                       0.43801613, -0.83891018,  0.72300822,  0.42764385,  0.48444319,
                                                       0.34606469, -0.31548898,  0.34424614,  0.07873913, -0.91834060,
                                                      -0.17023571,  0.15497373, -1.42028829, -0.50732354,  0.38278548,
                                                      -0.63573374, -0.53303308,  0.27223804, -0.20991905,  0.45978979,
                                                      -0.60859952,  0.78597102, -0.31466016, -0.50330494,  0.07287572,
                                                       1.30384166, -0.15242888,  0.32414952,  0.99723693, -1.27655555,
                                                      -1.21717734});
            t0 = t0.reshape(shape);
            int[] axis = { 1 };
            var expectedResult = new Tensor<double>(new double[] { 1.87176095, 2.04972621, -1.28993997, -1.87195152, 1.0938915, -1.22275325 });
            int[] expectedShape = { 3, 2 };
            expectedResult = expectedResult.reshape(expectedShape);

            // Act
            var sum = t0.sum(axis);

            // Assert
            int[] actualShape = sum.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing sum");
            for (int i = 0; i < t0.Shape[0]; i++)
                for (int j = 0; j < t0.Shape[2]; j++)
                    Assert.AreEqual(expectedResult[i, j], sum[i, j], 1e-8, "Failed to determine correct sum value");
        }

        [TestMethod]
        public void Sum_MultipleAxes_CheckResult()
        {
            // Arrange
            int[] shape = { 3, 6, 2 };
            var t0 = new Tensor<double>(new double[] {-0.06314949,  0.19945749,  0.44094216,  0.52029016,  1.55916992,
                                                       0.43801613, -0.83891018,  0.72300822,  0.42764385,  0.48444319,
                                                       0.34606469, -0.31548898,  0.34424614,  0.07873913, -0.91834060,
                                                      -0.17023571,  0.15497373, -1.42028829, -0.50732354,  0.38278548,
                                                      -0.63573374, -0.53303308,  0.27223804, -0.20991905,  0.45978979,
                                                      -0.60859952,  0.78597102, -0.31466016, -0.50330494,  0.07287572,
                                                       1.30384166, -0.15242888,  0.32414952,  0.99723693, -1.27655555,
                                                      -1.21717734});
            t0 = t0.reshape(shape);
            int[] axes = { 0, 1 };
            var expectedResult = new Tensor<double>(new double[] { 1.67571248, -1.04497856 });
            int[] expectedShape = { 2 };

            // Act
            var sum = t0.sum(axes);

            // Assert
            int[] actualShape = sum.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to determine shape correctly during computing sum");
            for (int i=0; i< t0.Shape[2]; i++)
                Assert.AreEqual(expectedResult[i], sum[i], 1e-8, "Failed to determine correct sum value");
        }


        [TestMethod]
        public void MatrixMultiply_Multiply2Dx2D_CheckResultShapeAndValues()
        {
            // Arrange
            int[] shape0 = { 3, 4 };
            int[] shape1 = { 4, 5 };
            var t0 = new Tensor<double>(new double[] { 1.09939681,  1.93037217, -0.39616791, -0.8181301,
                                                      -0.10941912, -1.14258265, -0.13319872, -0.59856143, 
                                                      -0.02770170,  1.22942952,  0.50605386,  1.24099901});
            t0 = t0.reshape(shape0);
            var t1 = new Tensor<double>(new double[] { 0.71046264, -0.58194029, -0.91289933, -0.39884044,  0.98687941,
                                                      -0.12700939,  0.71551897, -1.69668304, -1.0853282 , -0.72358042,
                                                      -0.17675108, -0.59027432, -1.29883168, -0.38362994,  1.24856532,
                                                       0.52814952,  1.47486886,  1.15533313, -0.98225652,  0.40111821});
            t1 = t1.reshape(shape1);
            int[] expectedShape = { 3, 5 };
            var expectedResult = new Tensor<double>(new double[] { 0.17383305, -0.23135226, -4.70952571, -1.57797576, -1.13461582,
                                                                  -0.22520639, -1.55804000,  1.51995411,  1.92275783,  0.31236577,
                                                                   0.39015735,  2.42740108, -1.28417487, -2.73640275,  0.21269924});
            expectedResult = expectedResult.reshape(expectedShape);

            // Act
            var result = t0.matrixMultiply(t1);

            // Assert
            int[] actualShape = result.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to produce matrix product with correct shape (2D x 2D)");
            for (int i = 0; i < expectedShape[0]; i++)
                for (int j = 0; j < expectedShape[1]; j++)
                    Assert.AreEqual(expectedResult[i, j], result[i, j], 1e-8, "Failed to produce correct value for matrix product (2D x 2D)");
        }

        [TestMethod]
        public void ScalarMulitply_MultiplyND_CheckResultShapeAndValue()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {0.52418050, -1.81009897, 0.76947064, -1.19900647,  0.22721830,
                                                      0.77415511, -0.18234184, 0.64216762,  0.94072688, -0.35556122,
                                                      0.14524260, -2.33428278, 2.18217591, -0.67387352,  0.33823934});
            t0 = t0.reshape(shape);
            double factor = -0.293;
            var expectedResult = new Tensor<double>(new double[] {-0.15358489, 0.53035900, -0.22545490,  0.35130890, -0.06657496,
                                                                  -0.22682745, 0.05342616, -0.18815511, -0.27563298,  0.10417944,
                                                                  -0.04255608, 0.68394485, -0.63937754,  0.19744494, -0.09910413});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.scalarMultiply(factor);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for product with scalar");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for product with a scalar");
        }

        [TestMethod]
        public void ScalarMulitply_MultiplyNDInplace_CheckResultShapeAndValueAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {0.52418050, -1.81009897, 0.76947064, -1.19900647,  0.22721830,
                                                      0.77415511, -0.18234184, 0.64216762,  0.94072688, -0.35556122,
                                                      0.14524260, -2.33428278, 2.18217591, -0.67387352,  0.33823934});
            t0 = t0.reshape(shape);
            double factor = -0.293;
            var expectedResult = new Tensor<double>(new double[] {-0.15358489, 0.53035900, -0.22545490,  0.35130890, -0.06657496,
                                                                  -0.22682745, 0.05342616, -0.18815511, -0.27563298,  0.10417944,
                                                                  -0.04255608, 0.68394485, -0.63937754,  0.19744494, -0.09910413});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.scalarMultiply(factor, inplace: true);
            t0[2, 4] = -8.5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for product with scalar");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems-1; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for product with a scalar");
            Assert.AreEqual(t0[2, 4], result[2,4], 1e-8, "Failed to override data inplace");
        }

        [TestMethod]
        public void ScalarMulitply_MultiplyNDNotInplace_CheckResultShapeAndValueAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {0.52418050, -1.81009897, 0.76947064, -1.19900647,  0.22721830,
                                                      0.77415511, -0.18234184, 0.64216762,  0.94072688, -0.35556122,
                                                      0.14524260, -2.33428278, 2.18217591, -0.67387352,  0.33823934});
            t0 = t0.reshape(shape);
            double factor = -0.293;
            var expectedResult = new Tensor<double>(new double[] {-0.15358489, 0.53035900, -0.22545490,  0.35130890, -0.06657496,
                                                                  -0.22682745, 0.05342616, -0.18815511, -0.27563298,  0.10417944,
                                                                  -0.04255608, 0.68394485, -0.63937754,  0.19744494, -0.09910413});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.scalarMultiply(factor, inplace: false);
            t0[2, 4] = -8.5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for product with scalar");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for product with a scalar");
        }


        [TestMethod]
        public void Dot_Multiply2Dx2D_CheckResultShapeAndValues()
        {
            // Arrange
            int[] shape0 = { 3, 4 };
            int[] shape1 = { 4, 5 };
            var t0 = new Tensor<double>(new double[] { 1.09939681,  1.93037217, -0.39616791, -0.8181301,
                                                      -0.10941912, -1.14258265, -0.13319872, -0.59856143,
                                                      -0.02770170,  1.22942952,  0.50605386,  1.24099901});
            t0 = t0.reshape(shape0);
            var t1 = new Tensor<double>(new double[] { 0.71046264, -0.58194029, -0.91289933, -0.39884044,  0.98687941,
                                                      -0.12700939,  0.71551897, -1.69668304, -1.0853282 , -0.72358042,
                                                      -0.17675108, -0.59027432, -1.29883168, -0.38362994,  1.24856532,
                                                       0.52814952,  1.47486886,  1.15533313, -0.98225652,  0.40111821});
            t1 = t1.reshape(shape1);
            int[] expectedShape = { 3, 5 };
            var expectedResult = new Tensor<double>(new double[] { 0.17383305, -0.23135226, -4.70952571, -1.57797576, -1.13461582,
                                                                  -0.22520639, -1.55804000,  1.51995411,  1.92275783,  0.31236577,
                                                                   0.39015735,  2.42740108, -1.28417487, -2.73640275,  0.21269924});
            expectedResult = expectedResult.reshape(expectedShape);

            // Act
            var result = t0.dot(t1);

            // Assert
            int[] actualShape = result.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to produce dot product with correct shape (2D x 2D)");
            for (int i = 0; i < expectedShape[0]; i++)
                for (int j = 0; j < expectedShape[1]; j++)
                    Assert.AreEqual(expectedResult[i, j], result[i, j], 1e-8, "Failed to produce correct values for dot product (2D x 2D)");

        }

        [TestMethod]
        public void Dot_MultiplyNDx0D_CheckResultShapeAndValues()
        {
            // Arrange
            double scalar = 5.7;
            var t0 = new Tensor<double>(new double[] {0.12724366,  1.69624426, -1.73220899,
                                                      1.46975827, -0.19054357,  0.25061233,
                                                     -0.69244312,  1.09303574,  0.51491963,
                                                     -0.65492861, -0.53312317,  0.72893695,
                                                      0.38765699, -1.33712305, -0.58853967});
            int[] shape = { 5, 3 };
            t0 = t0.reshape(shape);

            // Act
            var result = t0.dot(scalar);

            // Assert
            int[] actualShape = result.Shape;
            Assert.AreEqual(shape, actualShape, "Failed to produce dot product with correct shape(2D x 0D)");
            for (int i = 0; i < shape[0]; i++)
                for (int j = 0; j < shape[1]; j++)
                    Assert.AreEqual(t0[i, j] * scalar, result[i, j], 1e-8, "Failed to produce correct value for dot product (2D x 0D)");
        }


        [TestMethod]
        public void Dot_Multiply1Dx1D_CheckResultShapeAndValues()
        {
            // Arrange 
            var t0 = new Tensor<double>(new double[] {0.97128908, 0.20534807, 0.52669755, 
                                                      0.83773758, 0.99046507, 0.53007248, 
                                                      0.66714638, 0.79524998, 0.54611307});
            var t1 = new Tensor<double>(new double[] {0.3557552 , 0.72256437, 0.12421650,
                                                      0.95843971, 0.0366442 , 0.84412559,
                                                      0.72925096, 0.99252063, 0.79177809});
            double expectedValue = 3.554225882;
            int[] expectedShape = new int[0];

            // Act
            var result = t0.dot(t1);

            // Assert
            var actualShape = result.Shape;
            var actualValue = result[0];
            CollectionAssert.AreEqual(expectedShape, actualShape);
            Assert.AreEqual(expectedValue, actualValue, 1e-8, "Failed to produce correct value for dot product (1D x 1D)");
        }

        [TestMethod]
        public void Dot_MultiplyNDx1D_CheckResultShapeAndVales()
        {
            // Arrange
            int[] shape0 = { 3, 3, 4 };
            var t0 = new Tensor<double>(new double[] { 0.10926628, -1.18684222, -0.57741418, -0.89062215,
                                                       0.84142014, -0.50157443, -0.57049617,  0.62246595, 
                                                       0.44615705,  0.79589776, -1.15958190,  0.36160481, 
                                                       0.54333688,  0.54801840, -1.05182413,  0.70247984, 
                                                      -0.32466510,  0.23987538, -0.65490072,  0.08786857,
                                                       2.21031957, -0.07222872,  0.47664952, -1.20965737,
                                                       0.62392825,  0.24664832,  1.28666002, -0.03601034, 
                                                       0.06877738, -0.41008982, -0.51763828,  1.55661661, 
                                                      -1.35144804, -0.10888638,  0.16809589, -0.29611745});
            t0 = t0.reshape(shape0);
            var t1 = new Tensor<double>(new double[] { 1.59656932, -0.59763728, -1.06721526, -0.30083897});
            int[] expectedShape = { 3, 3 };
            var expectedResult = new Tensor<double>(new double[] { 1.76791142, 2.06472536,  1.36540117,
                                                                   1.45114822, 0.01077694,  3.42731942,
                                                                  -0.51357142, 0.43903335, -2.18291675});
            expectedResult = expectedResult.reshape(expectedShape);

            // Act
            var result = t0.dot(t1);

            // Assert
            var actualShape = result.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, "Failed to produce dot product with correct shape (ND x 1D)");
            for (int i = 0; i < expectedShape[0]; i++)
                for (int j = 0; j < expectedShape[1]; j++)
                    Assert.AreEqual(expectedResult[i, j], result[i, j], 1e-8, "Failed to produce correct value for dot product (ND x 1D)");
        }

        [TestMethod]
        public void Dot_MultiplyNDxMD_CheckResultShapeAndValues()
        {
            // Arrange
            int[] shape0 = { 10, 9, 8 };
            int[] shape1 = { 17, 8, 11 };
            var t0 = Tensor<double>.randNormal(0, 1, shape0);
            var t1 = Tensor<double>.randNormal(0, 1, shape1);
            int[] expectedShape = { 10, 9, 17, 11 };

            // Act
            var result = t0.dot(t1);

            // Assert
            int[] actualShape = result.Shape;
            CollectionAssert.AreEqual(expectedShape, actualShape, 
                "Failed to produce dot product with correct shape (ND x MD)");
            for (int i = 0; i < t0.Shape[0]; i++)
                for (int j = 0; j < t0.Shape[1]; j++)
                    for (int k = 0; k < t1.Shape[0]; k++)
                        for (int l = 0; l < t1.Shape[2]; l++)
                        {
                            var t0Slice = t0[i..(i + 1), j..(j + 1), 0..^0].reshape(1, shape0[2]);
                            var t1Slice = t1[k..(k + 1), 0..^0, l..(l + 1)].reshape(shape0[2], 1);
                            var productSlices = t0Slice.matrixMultiply(t1Slice)[0, 0];

                            Assert.AreEqual(result[i, j, k, l], productSlices, 1e-8,
                                "Failed to produce correct values for dot product (ND x MD)");
                        }
        }


        [TestMethod]
        public void Add_NDxND_CheckResultShapeAndValues()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {1.03227437,  0.10694046, -0.32972069, -0.00646071,
                                                                  1.71438576,  0.35177967, -1.12324639,  3.96297024,
                                                                  2.25853041, -0.06646940,  1.06258025, -0.68143846});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.add(t1);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise sum");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise sum");
        }

        [TestMethod]
        public void Add_NDxNDInplace_CheckResultShapeAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {1.03227437,  0.10694046, -0.32972069, -0.00646071,
                                                                  1.71438576,  0.35177967, -1.12324639,  3.96297024,
                                                                  2.25853041, -0.06646940,  1.06258025, -0.68143846});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.add(t1, inplace: true);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise sum");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            Assert.AreEqual(t0[0, 0], result[0, 0], 1e-8, "Failed to override data inplace");
            for (int i = 1; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise sum");
        }

        [TestMethod]
        public void Add_NDxNDNotInplace_CheckResultShapeAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {1.03227437,  0.10694046, -0.32972069, -0.00646071,
                                                                  1.71438576,  0.35177967, -1.12324639,  3.96297024,
                                                                  2.25853041, -0.06646940,  1.06258025, -0.68143846});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.add(t1);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise sum");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise sum");
        }


        [TestMethod]
        public void Subtract_NDxND_CheckResultShapeAndValues()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {-0.97088703, -1.0381924, -0.17020805, -0.16917945,
                                                                   0.76655132, -0.55445567, 1.27257101, -0.01317976,
                                                                   0.92748873, -0.5711765, -2.31951179,  0.17761286});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.subtract(t1);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise difference");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise difference");
        }

        [TestMethod]
        public void Subtract_NDxNDInplace_CheckResultShapeAndValuesAndInplaceNess()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {-0.97088703, -1.0381924, -0.17020805, -0.16917945,
                                                                   0.76655132, -0.55445567, 1.27257101, -0.01317976,
                                                                   0.92748873, -0.5711765, -2.31951179,  0.17761286});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.subtract(t1, inplace: true);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise difference");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            Assert.AreEqual(t0[0, 0], result[0, 0], 1e-8, "Failed to override data inplace");
            for (int i = 1; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise difference");
        }

        [TestMethod]
        public void Subtract_NDxNDNotInplace_CheckResultShapeAndValuesAndInplaceNess()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {-0.97088703, -1.0381924, -0.17020805, -0.16917945,
                                                                   0.76655132, -0.55445567, 1.27257101, -0.01317976,
                                                                   0.92748873, -0.5711765, -2.31951179,  0.17761286});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.subtract(t1);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise difference");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise difference");
        }

        [TestMethod]
        public void Multiply_NDxND_CheckResultShapeAndValues()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {0.03074219, -0.26660180,  0.01993624, -0.00714499,
                                                                  0.58787940, -0.04591804, -0.08943863,  3.92623985,
                                                                  1.06018107, -0.08045610, -1.06276454,  0.10820301});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.multiply(t1);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise product");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise product");
        }

        [TestMethod]
        public void Multiply_NDxNDInplace_CheckResultShapeAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {0.03074219, -0.26660180,  0.01993624, -0.00714499,
                                                                  0.58787940, -0.04591804, -0.08943863,  3.92623985,
                                                                  1.06018107, -0.08045610, -1.06276454,  0.10820301});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.multiply(t1, inplace: true);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise product");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            Assert.AreEqual(t0[0, 0], result[0, 0], 1e-8, "Failed to override data inplace");
            for (int i = 1; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise product");
        }

        [TestMethod]
        public void Multiply_NDxNDNotInplace_CheckResultShapeAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {0.03074219, -0.26660180,  0.01993624, -0.00714499,
                                                                  0.58787940, -0.04591804, -0.08943863,  3.92623985,
                                                                  1.06018107, -0.08045610, -1.06276454,  0.10820301});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.multiply(t1);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise product");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise product");
        }

        [TestMethod]
        public void Divide_NDxND_CheckResultShapeAndValues()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {0.03064523, -0.81322611,  3.1341011, -1.07940954,
                                                                  2.61747936, -0.22364610, -0.06232721, 0.99337059,
                                                                  2.39362838, -1.26339792, -0.37164321, 0.58649069});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.divide(t1);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise quotient");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise quotient");
        }

        [TestMethod]
        public void Divide_NDxNDInplace_CheckResultShapeAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {0.03064523, -0.81322611,  3.1341011, -1.07940954,
                                                                  2.61747936, -0.22364610, -0.06232721, 0.99337059,
                                                                  2.39362838, -1.26339792, -0.37164321, 0.58649069});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.divide(t1, inplace: true);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise quotient");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            Assert.AreEqual(t0[0, 0], result[0, 0], 1e-8, "Failed to override data inplace");
            for (int i = 1; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise quotient");
        }

        [TestMethod]
        public void Divide_NDxNDNotInplace_CheckResultShapeAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 4 };
            var t0 = new Tensor<double>(new double[] {0.03069367, -0.46562597, -0.24996437, -0.08782008,
                                                      1.24046854, -0.10133800,  0.07466231,  1.97489524,
                                                      1.59300957, -0.31882295, -0.62846577, -0.2519128});
            t0 = t0.reshape(shape);
            var t1 = new Tensor<double>(new double[] {1.00158070,  0.57256643, -0.07975632,  0.08135937,
                                                      0.47391722,  0.45311767, -1.19790870,  1.98807500,
                                                      0.66552084,  0.25235355,  1.69104602, -0.42952566});
            t1 = t1.reshape(shape);

            var expectedResult = new Tensor<double>(new double[] {0.03064523, -0.81322611,  3.1341011, -1.07940954,
                                                                  2.61747936, -0.22364610, -0.06232721, 0.99337059,
                                                                  2.39362838, -1.26339792, -0.37164321, 0.58649069});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.divide(t1);
            t0[0, 0] = 5.0;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise quotient");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise quotient");
        }


        [TestMethod]
        public void Exp_ND_CheckResultsAndValues()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] {0.41989324, 0.56735334, 1.42530178, 2.01073965, 0.29185243,
                                                                  0.43047123, 3.32072297, 0.59259754, 0.66366033, 0.24148564,
                                                                  1.98610914, 1.59572044, 1.41667814, 2.52283157, 7.89325171});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.exp();

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void Exp_NDInplace_CheckResultsAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] {0.41989324, 0.56735334, 1.42530178, 2.01073965, 0.29185243,
                                                                  0.43047123, 3.32072297, 0.59259754, 0.66366033, 0.24148564,
                                                                  1.98610914, 1.59572044, 1.41667814, 2.52283157, 7.89325171});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.exp(inplace: true);
            t0[0, 0] = 5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            Assert.AreEqual(t0[0, 0], result[0, 0], 1e-8, "Failed to override data inplace");
            for (int i = 1; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void Exp_NDNotInplace_CheckResultsAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] {0.41989324, 0.56735334, 1.42530178, 2.01073965, 0.29185243,
                                                                  0.43047123, 3.32072297, 0.59259754, 0.66366033, 0.24148564,
                                                                  1.98610914, 1.59572044, 1.41667814, 2.52283157, 7.89325171});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.exp();
            t0[0, 0] = 5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void NegExp_ND_CheckResultsAndValues()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] { 2.38155775, 1.76257003, 0.70160580, 0.49732943, 3.42638915,
                                                                   2.32303563, 0.30113924, 1.68748592, 1.50679490, 4.14103303,
                                                                   0.50349700, 0.62667619, 0.70587664, 0.39638001, 0.12669050});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.negExp();

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void NegExp_NDInplace_CheckResultsAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] { 2.38155775, 1.76257003, 0.70160580, 0.49732943, 3.42638915,
                                                                   2.32303563, 0.30113924, 1.68748592, 1.50679490, 4.14103303,
                                                                   0.50349700, 0.62667619, 0.70587664, 0.39638001, 0.12669050});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.negExp(inplace: true);
            t0[0, 0] = 5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            Assert.AreEqual(t0[0, 0], result[0, 0], 1e-8, "Failed to override data inplace");
            for (int i = 1; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void NegExp_NDNotInplace_CheckResultsAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] { 2.38155775, 1.76257003, 0.70160580, 0.49732943, 3.42638915,
                                                                   2.32303563, 0.30113924, 1.68748592, 1.50679490, 4.14103303,
                                                                   0.50349700, 0.62667619, 0.70587664, 0.39638001, 0.12669050});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.negExp();
            t0[0, 0] = 5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void Abs_ND_CheckResultsAndValues()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] {0.86775479, 0.56677299, 0.35438357, 0.69850264, 1.23150698,
                                                                  0.84287479, 1.20018252, 0.52323980, 0.40998481, 1.42094528,
                                                                  0.68617752, 0.46732532, 0.34831479, 0.92538191, 2.06600818});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.abs();

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void Abs_NDInplace_CheckResultsAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] {0.86775479, 0.56677299, 0.35438357, 0.69850264, 1.23150698,
                                                                  0.84287479, 1.20018252, 0.52323980, 0.40998481, 1.42094528,
                                                                  0.68617752, 0.46732532, 0.34831479, 0.92538191, 2.06600818});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.abs(inplace: true);
            t0[0, 0] = 5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            Assert.AreEqual(t0[0, 0], result[0, 0], 1e-8, "Failed to override data inplace");
            for (int i = 1; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }

        [TestMethod]
        public void Abs_NDNotInplace_CheckResultsAndValuesAndInplaceness()
        {
            // Arrange
            int[] shape = { 3, 5 };
            var t0 = new Tensor<double>(new double[] {-0.86775479, -0.56677299,  0.35438357,  0.69850264, -1.23150698,
                                                      -0.84287479,  1.20018252, -0.52323980, -0.40998481, -1.42094528,
                                                       0.68617752,  0.46732532,  0.34831479,  0.92538191,  2.06600818});
            t0 = t0.reshape(shape);
            var expectedResult = new Tensor<double>(new double[] {0.86775479, 0.56677299, 0.35438357, 0.69850264, 1.23150698,
                                                                  0.84287479, 1.20018252, 0.52323980, 0.40998481, 1.42094528,
                                                                  0.68617752, 0.46732532, 0.34831479, 0.92538191, 2.06600818});
            expectedResult = expectedResult.reshape(shape);

            // Act
            var result = t0.abs();
            t0[0, 0] = 5;

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to produce correct shape for elementwise exponentiation");
            var tmp0 = expectedResult.flatten();
            var tmp1 = result.flatten();
            for (int i = 0; i < t0.NumElems; i++)
                Assert.AreEqual(tmp0[i], tmp1[i], 1e-8, "Failed to produce correct values for elementwise exponentiation");
        }


        [TestMethod]
        public void Arange_OnePositiveIntegerArgument_ShouldCreateIntegerRange()
        {
            // Arrange
            int upperBound = 3;
            var expectedResult = new Tensor<double>(new double[] { 0, 1, 2 });

            // Act
            var result = Tensor<double>.arange(upperBound);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to create range Tensor with correct shape");
            for (int i = 0; i < expectedResult.NumElems; i++)
                Assert.AreEqual(expectedResult[i], result[i], 1e-8, "Failed to create range Tensor with correct values");
        }

        [TestMethod]
        public void Arange_OnePositiveNonIntegerArgument_ShouldCreateFlooredIntegerRange()
        {
            // Arrange
            float upperBound = 5.3f;
            var expectedResult = new Tensor<float>(new float[] { 0, 1, 2, 3, 4, 5 });

            // Act
            var result = Tensor<float>.arange(upperBound);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to create range Tensor with correct shape");
            for (int i = 0; i < expectedResult.NumElems; i++)
                Assert.AreEqual(expectedResult[i], result[i], 1e-8, "Failed to create range Tensor with correct values");
        }

        [TestMethod]
        public void Arange_PositiveRangeFractionalStep_ShouldCreateFractionalRange()
        {
            // Arrange
            double lowerBound = -1.0;
            double upperBound = 7.3;
            double increment = 0.7;
            var expectedResult = new Tensor<double>(new double[] { -1.0, -0.3, 0.4, 1.1, 1.8, 2.5, 3.2, 3.9, 4.6, 5.3, 6.0, 6.7 });

            // Act
            var result = Tensor<double>.arange(lowerBound, upperBound, increment);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to create range Tensor with correct shape");
            for (int i = 0; i < expectedResult.NumElems; i++)
                Assert.AreEqual(expectedResult[i], result[i], 1e-8, "Failed to create range Tensor with correct values");
        }

        [TestMethod]
        public void Arange_NegativeRangePositiveUnitStep_ShouldCreateEmptyTensor()
        {
            // Arange
            double lowerBound = 5;
            double upperBound = 3;
            int[] expectedShape = { 0 };
            int expectedNumElems = 0;

            // Act
            var result = Tensor<double>.arange(lowerBound, upperBound);

            // Assert
            CollectionAssert.AreEqual(expectedShape, result.Shape, "Failed to create range Tensor with correct shape");
            Assert.AreEqual(expectedNumElems, result.NumElems, "Failed to create range Tensor with correct number of elements");
        }

        public void Arange_NegativeRangeNegativeStep_ShouldCreateRevertedRange()
        {
            // Arrange
            double lowerBound = 5;
            double upperBound = -12;
            double increment = -1.5;
            var expectedResult = new Tensor<double>(new double[] { 5, 3.5, 2.0, 0.5, -1.0, -2.5, -4.0, -5.5, -7.0, -8.5, -10.0, -11.5 });

            // Act
            var result = Tensor<double>.arange(lowerBound, upperBound, increment);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to create linspace Tensor with correct shape");
            for (int i = 0; i < expectedResult.NumElems; i++)
                Assert.AreEqual(expectedResult[i], result[i], 1e-8, "Failed to create linspace Tensor with correct values");
        }


        [TestMethod]
        public void Linspace_PositiveRange_CheckResultShapeAndValues()
        {
            // Arrange
            double lowerBound = -3.0;
            double upperBound = 5.0;
            int num = 5;
            var expectedResult = new Tensor<double>(new double[] { -3.0, -1.0, 1.0, 3.0, 5.0 });

            // Act
            var result = Tensor<double>.linspace(lowerBound, upperBound, num);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to create linspace Tensor with correct shape");
            for (int i = 0; i < expectedResult.NumElems; i++)
                Assert.AreEqual(expectedResult[i], result[i], 1e-8, "Failed to create linspace Tensor with correct values");
        }

        [TestMethod]
        public void Linspace_NegativeRange_ShouldReturnRevertedRange()
        {
            // Arrange
            double lowerBound = 10.0;
            double upperBound = 6.5;
            int num = 8;
            var expectedResult = new Tensor<double>(new double[] { 10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5 });

            // Act
            var result = Tensor<double>.linspace(lowerBound, upperBound, num);

            // Assert
            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to create linspace Tensor with correct shape");
            for (int i = 0; i < expectedResult.NumElems; i++)
                Assert.AreEqual(expectedResult[i], result[i], 1e-8, "Failed to create linspace Tensor with correct values");
        }

        [TestMethod]
        public void Linspace_EmptyRange_ShouldReturnConstantRange()
        {
            // Arranage
            double lowerBound = 3.0;
            int num = 8;
            var expectedResult = new Tensor<double>(new double[] { 3, 3, 3, 3, 3, 3, 3, 3 });

            // Act
            var result = Tensor<double>.linspace(lowerBound, lowerBound, num);

            // Assert
            CollectionAssert.AreEqual(expectedResult.Shape, result.Shape, "Failed to create linspace Tensor with correct shape");
            for (int i = 0; i < expectedResult.NumElems; i++)
                Assert.AreEqual(expectedResult[i], result[i], 1e-8, "Failed to create linspace Tensor with correct values");
        }
    }
}
