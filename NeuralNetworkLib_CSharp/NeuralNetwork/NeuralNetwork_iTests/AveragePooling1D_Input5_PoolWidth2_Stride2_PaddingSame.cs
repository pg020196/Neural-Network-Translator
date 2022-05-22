using System;
using System.Collections.Generic;
using NeuralNetwork;
using Layers;
using Tensor;

namespace NeuralNetwork //_iTests
{
	public class AveragePooling1D_Input5_PoolWidth2_Stride2_PaddingSame
    {
        private readonly int myNumLayers;
        private readonly int[] myLayerOutputHeight;
        private readonly int[] myLayerOutputWidth;
        private readonly int[] myLayerOutputDepth;
        private readonly int[] myLayerTypes;
        private readonly int[] myActivationTypes;
        private readonly Tensor<double> myWeights;
        private readonly int[] myIndicesWeights;
        private readonly Tensor<double> myBias;
        private readonly int[] myIndicesBias;
        private readonly int[] myUseBias;
        private readonly int[] myPoolHeights;
        private readonly int[] myPoolWidths;
        private readonly int[] myVerticalStride;
        private readonly int[] myHorizontalStride;
        private readonly int[] myPadding;

        public NeuralNetwork<double> NeuralNetwork { get; }

        static private ActivationType DecodeActivationType(int code)
        {
            switch (code)
            {
                case (0):
                    return ActivationType.linear;
                case (1):
                    return ActivationType.sigmoid;
                case (2):
                    return ActivationType.relu;
                case (3):
                    return ActivationType.tanh;
                case (4):
                    return ActivationType.softmax;
                default:
                    throw new NotImplementedException();
            }
        }

        public AveragePooling1D_Input5_PoolWidth2_Stride2_PaddingSame()
        {
            myNumLayers = 1;
            myLayerOutputHeight = new int[] {5,3};
            myLayerOutputWidth = new int[] {1,1};
            myLayerOutputDepth = new int[] {0,1};
            myLayerTypes = new int[] {4};
            myActivationTypes = new int[] {0};
            myWeights = new Tensor<double>(new double[] {});
            myIndicesWeights = new int[] {0};
            myBias = new Tensor<double>(new double[] {});
            myIndicesBias = new int[] {0};
            myUseBias = new int[] {0};
            myPoolHeights = new int[] {2};
            myPoolWidths = new int[] {0};
            myVerticalStride = new int[] {2};
            myHorizontalStride = new int[] {1};
            myPadding = new int[] {1};

            int inputHeight = myLayerOutputHeight[0];
            int inputWidth = myLayerOutputWidth[0];
            int inputDepth = myLayerOutputDepth[0];

            int[] inputShape;
            if (inputWidth == 0)
            {
                 inputShape = new int[] { inputHeight };
            }
            else if (inputDepth == 0)
            {
                inputShape = new int[] { inputHeight, inputWidth };
            }
            else
            {
                inputShape = new int[] { inputHeight, inputWidth, inputDepth };
            }

            var inputLayer = new InputLayer<double>(inputShape);

            var layerList = new List<BaseLayer<double>>();
            layerList.Add(inputLayer);

            BaseLayer<double> previousLayer = inputLayer;

            for (int layerIndex = 1; layerIndex <= myNumLayers; layerIndex++)
            {
                switch (myLayerTypes[layerIndex - 1])
                {
                    case (1):
                        var denseLayer = CreateDenseLayer(layerIndex, previousLayer);
                        layerList.Add(denseLayer);
                        previousLayer = denseLayer;
                        break;
                    case (2): // flatten
                        var flattenLayer = new Flatten<double>(previousLayer.OutputShape);
                        layerList.Add(flattenLayer);
                        previousLayer = flattenLayer;
                        break;
                    case (3): // maxpooling
                        var maxPoolingLayer = CreateMaxPoolingLayer(layerIndex, previousLayer);
                        layerList.Add(maxPoolingLayer);
                        previousLayer = maxPoolingLayer;
                        break;
                    case (4): // avgpooling
                        var averagePoolingLayer = CreateAveragePoolingLayer(layerIndex, previousLayer);
                        layerList.Add(averagePoolingLayer);
                        previousLayer = averagePoolingLayer;
                        break;
                    default:
                        throw new NotImplementedException($"Layer type {myLayerTypes[layerIndex - 1]} not implemented");
                }
            }

            NeuralNetwork = new NeuralNetwork<double>(layerList);
        }

        private Dense<double> CreateDenseLayer(int layerIndex, BaseLayer<double> prevLayer)
        {
            int weightsIndexUpper;
            int biasIndexUpper;
            if (layerIndex == myNumLayers)
            {
                weightsIndexUpper = myWeights.Shape[0];
                biasIndexUpper = myBias.Shape[0];
            }
            else
            {
                weightsIndexUpper = myIndicesWeights[layerIndex];
                biasIndexUpper = myIndicesBias[layerIndex];
            }

            var activationType = DecodeActivationType(myActivationTypes[layerIndex - 1]);
            var useBias = Convert.ToBoolean(myUseBias[layerIndex - 1]);
            var denseLayer = new Dense<double>(prevLayer.OutputShape, myLayerOutputHeight[layerIndex], activationType, useBias);

            int[] weightsShape = { prevLayer.OutputShape[0], myLayerOutputHeight[layerIndex] };
            var layerWeights = myWeights[myIndicesWeights[layerIndex - 1]..weightsIndexUpper];
            layerWeights = layerWeights.reshape(weightsShape);

            int biasShape = myLayerOutputHeight[layerIndex];
            var layerBias = myBias[myIndicesBias[layerIndex - 1]..biasIndexUpper];
            layerBias = layerBias.reshape(biasShape);

            denseLayer.Weights = layerWeights;
            denseLayer.Bias = layerBias;

            return denseLayer;
        }

        private BaseLayer<double> CreateMaxPoolingLayer(int layerIndex, BaseLayer<double> previousLayer)
        {
            var poolHeight = myPoolHeights[layerIndex - 1];
            var poolWidth = myPoolWidths[layerIndex - 1];
            var is1dPooling = poolHeight != 0 && poolWidth == 0;
            var is2dPooling = poolHeight != 0 && poolWidth != 0;

            PaddingType paddingType;
            if (myPadding[layerIndex - 1] == 0)
            {
                paddingType = PaddingType.valid;
            }
            else
            {
                paddingType = PaddingType.same_keras;
            }

            if (is1dPooling)
            {
                int stride = myVerticalStride[layerIndex - 1];
                var maxPoolingLayer = new PoolingLayer1D<double>(previousLayer.OutputShape, PoolingType.max, poolHeight, stride, paddingType);
                return maxPoolingLayer;
            }
            else if (is2dPooling)
            {
                int[] stride = { myVerticalStride[layerIndex - 1], myHorizontalStride[layerIndex - 1] };
                int[] poolSize = { poolHeight, poolWidth };
                var maxPoolingLayer = new PoolingLayer2D<double>(previousLayer.OutputShape, PoolingType.max, poolSize, stride, paddingType);
                return maxPoolingLayer;
            }
            else
            {
                throw new NotSupportedException($"Layer {layerIndex} is average pooling, but pool height and width are zero.");
            }
        }

        private BaseLayer<double> CreateAveragePoolingLayer(int layerIndex, BaseLayer<double> previousLayer)
        {
            var poolHeight = myPoolHeights[layerIndex - 1];
            var poolWidth = myPoolWidths[layerIndex - 1];
            var is1dPooling = poolHeight != 0 && poolWidth == 0;
            var is2dPooling = poolHeight != 0 && poolWidth != 0;

            PaddingType paddingType;
            if (myPadding[layerIndex - 1] == 0)
            {
                paddingType = PaddingType.valid;
            }
            else
            {
                paddingType = PaddingType.same_keras;
            }

            if (is1dPooling)
            {
                int stride = myVerticalStride[layerIndex - 1];
                var averagePoolinglayer = new PoolingLayer1D<double>(previousLayer.OutputShape, PoolingType.average, poolHeight, stride, paddingType);
                return averagePoolinglayer;
            }
            else if (is2dPooling)
            {
                int[] stride = { myVerticalStride[layerIndex - 1], myHorizontalStride[layerIndex - 1] };
                int[] poolSize = { poolHeight, poolWidth };
                var averagePoolinglayer = new PoolingLayer2D<double>(previousLayer.OutputShape, PoolingType.average, poolSize, stride, paddingType);
                return averagePoolinglayer;
            }
            else
            {
                throw new NotSupportedException($"Layer {layerIndex} is average pooling, but pool height and width are zero.");
            }
        }
    }
}