using System;
//using System.Globalization;
//using System.Linq;
using System.Collections.Generic;
using Tensor;
using Layers;

namespace NeuralNetwork {

    public class NeuralNetwork<T>
    {
        // The Shape of the input to the neural network
        public int[] InputShape { get; private set; }
        public int[] OutputShape { get; private set; }

        public List<BaseLayer<T>> layers { get; private set; }

        public Tensor<T> predict(Tensor<T> input)
        {
            Tensor<T> output = input;
            for (int i = 0; i < layers.Count; i++)
                output = layers[i].FeedForward(output);
            return output;
        }

        //public void addLayer(BaseLayer<T> layer)
        //{
        //    layers.Add(layer);
        //}

        //public void compile()
        //{

        //}

        public NeuralNetwork(List<BaseLayer<T>> layers)
        {
            this.layers = layers;
            InputShape = layers[0].InputShape;
            OutputShape = layers[layers.Count - 1].OutputShape;
        }

    }

}