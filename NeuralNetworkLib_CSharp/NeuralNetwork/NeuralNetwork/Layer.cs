using System;
using System.Collections.Generic;
using System.Text;
using Tensor;
using Activation;


namespace Layers
{
    public enum LayerType
    {
        dense,
        flatten
    }

    public enum ActivationType
    {
        linear,
        softmax,
        relu,
        tanh,
        sigmoid
    }

    public enum InitializerType
    {
        zeros,
        randNormal,
        randUniform
    }

    public enum PoolingType
    {
        average,
        max
    }

    public enum PaddingType
    {
        valid,
        same,
        same_keras
    }

    //public enum FloatPrecision
    //{
    //    singlePrecision,
    //    doublePrecision
    //}

    public abstract class BaseLayer<T>
    {
        public int[] InputShape { get; protected set; }
        public int[] OutputShape { get; protected set; }
        public ActivationType ActivationType { get; protected set; }
        public abstract Tensor<T> FeedForward(Tensor<T> layer_input);

        protected void checkShapeConsistency(int[] currentInputShape, int[] layerInputShape)
        {
            for (int i = 1; i < currentInputShape.Length; i++)
            {
                if (currentInputShape[i] != layerInputShape[i - 1])
                {
                    var sb = new StringBuilder();
                    sb.Append("Wrong shape of tensor ");
                    sb.Append(IntExtension.arrayToString(currentInputShape));
                    sb.Append(",  must be of shape (?");
                    for (int j = 0; j < layerInputShape.Length; j++)
                    {
                        sb.Append(" x ");
                        sb.Append(layerInputShape[j]);
                    }
                    sb.Append(")");
                    throw new ArgumentException(sb.ToString());
                }
            }
        }

        // public abstract override String ToString();
    }


    public class InputLayer<T> : BaseLayer<T>
    {
        public InputLayer(int[] inputShape)
        {
            InputShape = inputShape;
            OutputShape = inputShape;
            ActivationType = ActivationType.linear;
        }

        public override Tensor<T> FeedForward(Tensor<T> input)
        {
            if (input.Dims < 2)
                throw new ArgumentException("Feature vectors cannot be zero dimensional, i.e. input must be at least of two dimensions");

            checkShapeConsistency(input.Shape, InputShape);

            return input;
        }
    }

    public class Dense<T> : BaseLayer<T>
    {
        private Tensor<T> weights;
        private Tensor<T> bias;

        // Tensors to hold the weights and biases of the layer
        public Tensor<T> Weights {
            get { return weights; }
            set
            {
                if (value.Dims != 2)
                    throw new ArgumentException("Weights of dense layer have to be 2D");
                if (value.Shape[0] != InputShape[0] || value.Shape[1] != NumUnits)
                    throw new ArgumentException("Weights must be of shape InputShape[0] x NumUnits");
                weights = value;
            }
        }
        public Tensor<T> Bias {
            get { return bias; }
            set {
                if (value.Dims != 1)
                    throw new ArgumentException("Bias of dense layer has to be 1D");
                if (value.Shape[0] != NumUnits)
                    throw new ArgumentException("Bias must be of shape NumUnits");
                bias = value;
            }
        }
        public int NumUnits { get; private set; }
        public bool UseBias { get; private set; }

        private IActivation<T> activation;

        public Dense(int[] inputShape, int numUnits, ActivationType activationType=ActivationType.linear, 
            bool useBias=true, InitializerType weightsInitializerType=InitializerType.randNormal, 
            double weightsInitializer0=0d, double weightsInitializer1 = 1d, 
            InitializerType biasInitializerType=InitializerType.randNormal,
            double biasInitializer0=0d, double biasInitializer1=1d) //, FloatPrecision prec=FloatPrecision.singlePrecision)
        {
            if (inputShape.Length != 1)
                throw new NotImplementedException("Currently only 1D samples to Dense layer supported");

            InputShape = inputShape;
            OutputShape = new int[1] { numUnits };
            ActivationType = activationType;

            switch (activationType)
            {
                case ActivationType.softmax:
                    activation = new Softmax<T>();
                    break;
                case ActivationType.relu:
                    activation = new ReLU<T>();
                    break;
                case ActivationType.tanh:
                    activation = new Tanh<T>();
                    break;
                case ActivationType.sigmoid:
                    activation = new Sigmoid<T>();
                    break;
                default:
                    activation = null;
                    break;
            }

            NumUnits = numUnits;
            UseBias = useBias;

            switch (weightsInitializerType)
            {
                case InitializerType.zeros:
                    Weights = Tensor<T>.zeros(inputShape[0], numUnits);
                    break;
                case InitializerType.randNormal:
                    Weights = Tensor<T>.randNormal(weightsInitializer0, weightsInitializer1, inputShape[0], numUnits);
                    break;
                case InitializerType.randUniform:
                    Weights = Tensor<T>.randNormal(weightsInitializer0, weightsInitializer1, inputShape[0], numUnits);
                    break;
                default:
                    throw new NotImplementedException();
            }

            if (useBias)
            {
                switch (biasInitializerType)
                {
                    case InitializerType.zeros:
                        Bias = Tensor<T>.zeros(numUnits);
                        break;
                    case InitializerType.randNormal:
                        Bias = Tensor<T>.randNormal(biasInitializer0, biasInitializer1, numUnits);
                        break;
                    case InitializerType.randUniform:
                        Bias = Tensor<T>.randUniform(biasInitializer0, biasInitializer1, numUnits);
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            else
                Bias = null;

        }

        public override Tensor<T> FeedForward(Tensor<T> input)
        {
            if (input.Dims != 2)
                throw new NotImplementedException("Currently only 1D samples supported");

            checkShapeConsistency(input.Shape, InputShape);

            //var result = Tensor<T>.zeros(input.Shape[0], NumUnits);
            var result = input.dot(Weights);
            if (UseBias)
            {
                var bias_augm = Tensor<T>.ones(input.Shape[0], 1).dot(Bias.reshape(1, NumUnits));
                result = result.add(bias_augm, inplace:true);
            }
            if (ActivationType != ActivationType.linear)
                result = activation.apply(result, inplace: true);
            return result;
        }
    }

    //public class MaxPooling<T> : BaseLayer<T>
    //{
    //    // padding size for padding type "same"
    //    // (Math.ceil(input_shape / strides) - 1) * strides + pool_size - input_shape
    //}


    class PoolingLayer1D<T> : BaseLayer<T>
    {
        public PaddingType paddingType { get; private set; }
        public PoolingType poolingType { get; private set; }
        public int pool_size { get; private set; }
        public int stride { get; private set; }

        private delegate T Pooling(Tensor<T> inputWindow);

        private T avgPoolingApply(Tensor<T> inputWindow)
        {
            if (inputWindow.Dims != 1)
            {
                throw new ArgumentException($"input must be 1D, but is of {inputWindow.Dims} dimensions");
            }

            return inputWindow.mean()[0];
        }

        private T maxPoolingApply(Tensor<T> inputWindow)
        {
            if (inputWindow.Dims != 1)
            {
                throw new ArgumentException($"input must be 1D, but is of {inputWindow.Dims} dimensions");
            }

            return inputWindow.max()[0];
        }


        public PoolingLayer1D(int[] inputShape, PoolingType poolingType, int pool_size, int stride = -1, PaddingType paddingType = PaddingType.valid)
        {
            if (inputShape.Length != 2)
            {
                string msg = $"inputShape must be 2D, but is of dimension {inputShape.Length}";
                throw new ArgumentException(msg);
            }
            InputShape = inputShape;

            this.poolingType = poolingType;
            switch (poolingType)
            {
                case PoolingType.average:
                    this.poolingApply = avgPoolingApply;
                    break;
                case PoolingType.max
            }
            

            this.paddingType = paddingType;

            if (pool_size < 1)
            {
                string msg = $"pool_size must be at least 1, but is {pool_size}";
                throw new ArgumentException(msg);
            }
            else if (paddingType == PaddingType.valid & pool_size > inputShape[0])
            {
                string msg = "pool_size cannot be larger than first dimension of input_shape for paddingType 'valid'";
                throw new ArgumentException(msg);
            }

            this.pool_size = pool_size;

            if (stride == -1)
            {
                stride = pool_size;
            }
            else if (stride < 1)
            {
                string msg = $"stride cannot be less than 1, but is {stride}";
                throw new ArgumentException(msg);
            }

            int outputShape_0;
            if (paddingType == PaddingType.valid)
            {
                outputShape_0 = (int) Math.Ceiling((double)(InputShape[0] - pool_size + 1) / stride);
            }
            else
            {
                outputShape_0 = (int) Math.Ceiling((double)InputShape[0] / stride);
            }

            OutputShape = new int[] { outputShape_0, inputShape[1] };
        }

        public override Tensor<T> FeedForward(Tensor<T> input)
        {
            checkShapeConsistency(input.Shape, InputShape);



            switch (paddingType)
            {
                case PaddingType.valid:
                    var output = Tensor<T>.zeros(OutputShape);
                    for (int i=0; i<input.Shape[0]; i++)
                    {
                        for (int j=0; j<input.Shape[2]; j++)
                        {
                            for (int k=0; k<OutputShape[0]; k++)
                            {
                                if ()
                                output[i, k, j] = ;
                            }
                        }
                    }

                    break;
                case PaddingType.same:
                    throw new NotImplementedException();
                    break;
                case PaddingType.same_keras:
                    throw new NotImplementedException();
                    break;
                default:
                    throw new NotImplementedException();
            }

        }


        static Tensor<T> padding(Tensor<T> input, int h0=0, int h1=0, int w0=0, int w1=0, T value=)
    }
}

