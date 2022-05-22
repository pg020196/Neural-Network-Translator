using System;
using Tensor;

namespace Activation
{
    public interface IActivation<T>
    {
        Tensor<T> apply(Tensor<T> t, bool inplace = true);
    }

    public class Tanh<T> : IActivation<T>
    { 
        public Tensor<T> apply(Tensor<T> t, bool inplace=true)
        {
            // tanh(x) = 1 - 2 / (exp(2x) + 1)
            var denominator = t.exp(inplace); // exp(x)
            denominator = denominator.multiply(denominator, inplace); // exp(2x) = exp(x) * exp(x)
            var one = Tensor<T>.ones(denominator.Shape); // 1 as Tensor of equal shape
            denominator = denominator.add(one, inplace); // 1 + exp(2x)
            //var fraction = one.divide(denominator);  // 1 / ( 1 + exp(2x) )
            var fraction = denominator.invert(inplace);
            fraction = fraction.add(fraction, inplace); // 2 / ( 1 + exp(2x) )
            //return one.subtract(fraction, inplace); // 1 - 2 / ( 1 + exp(2x) )
            return fraction.negate(inplace).add(one, inplace);
        }
    }

    public class ReLU<T> : IActivation<T>
    {
        public Tensor<T> apply(Tensor<T> t, bool inplace = true)
        {
            // relu(x) = ( x + abs(x) ) / 2
            return t.add(t.abs(inplace: false), inplace).scalarMultiply(0.5, inplace);
        }
    }

    public class Sigmoid<T> : IActivation<T>
    {
        public Tensor<T> apply(Tensor<T> t, bool inplace=true)
        {
            // sigmoid(x) = 1 / ( 1 + exp(-x) )
            var one = Tensor<T>.ones(t.Shape);
            //return one.divide(t.negExp(inplace).add(one, inplace), inplace);
            return t.negExp(inplace).add(one, inplace).invert(inplace);
        }
    }

    public class Softmax<T> : IActivation<T>
    {
        public Tensor<T> apply(Tensor<T> t, bool inplace = true)
        {
            //throw new NotImplementedException();
            if (t.Dims != 2)
                throw new ArgumentException("Cannot apply softmax activation to Tensor with other than two dimensions.");

            // Softmax(x) = exp(x) / sum(exp(x), axis=-1)
            var exp_t = t.exp(inplace);
            var denominator = exp_t.sum(axes: -1);
            denominator = denominator.reshape(denominator.Shape[0], 1).dot(Tensor<T>.ones(1, t.Shape[1]));
            return exp_t.divide(denominator, inplace: true);
        }
    }
}
