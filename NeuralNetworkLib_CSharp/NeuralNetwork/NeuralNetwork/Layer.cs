using System;
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

            for (int i = 1; i < input.Shape.Length; i++)
                if (input.Shape[i] != InputShape[i - 1])
                {
                    var sb = new StringBuilder();
                    sb.Append("Wrong shape of tensor ");
                    sb.Append(IntExtension.arrayToString(input.Shape));
                    sb.Append(",  must be of shape (?");
                    for (int j = 0; j < InputShape.Length; j++)
                    {
                        sb.Append(" x ");
                        sb.Append(InputShape[j]);
                    }
                    sb.Append(")");
                    throw new ArgumentException(sb.ToString());
                }
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

    public class MaxPooling<T> : BaseLayer<T>
    {
        // padding size for padding type "same"
        // (Math.ceil(input_shape / strides) - 1) * strides + pool_size - input_shape
    }



    class LayerUtils
    {
        static Tensor<T> 
    }
}

