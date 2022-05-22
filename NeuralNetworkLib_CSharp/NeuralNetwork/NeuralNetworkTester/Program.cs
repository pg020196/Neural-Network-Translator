using Layers;
using NeuralNetwork;
using System;
using System.Collections.Generic;
using Tensor;

namespace NeuralNetworkTester
{
    class Program
    {
        static void Main(string[] args)
        {
            //var inputLayer = new InputLayer<double>(new int[] { 10 });
            //var dense0 = new Dense<double>(inputLayer.OutputShape, 10, ActivationType.relu);
            //var dense1 = new Dense<double>(dense0.OutputShape, 5, ActivationType.softmax);

            //var layers = new List<BaseLayer<double>> { inputLayer, dense0, dense1 };

            //var net = new NeuralNetwork<double>(layers);

            //var x = Tensor<double>.randUniform(0, 1, 100, 10);
            //var y = net.predict(x);

            //var sum_y = y.sum(axes: 0);

            //Console.WriteLine(y);
            //Console.WriteLine(sum_y);


            ////////////////////////////////////////////////

            var networkBuilder = new NeuralNetworkBuilder.NeuralNetworkBuilder();
            var network = networkBuilder.NeuralNetwork;

            int batchSize = 1;
            var shape = new List<int>();
            shape.Add(batchSize);
            shape.AddRange(network.InputShape);
            var X = Tensor<double>.randNormal(0, 1, shape.ToArray());
            Console.WriteLine(X);

            var y = network.predict(X);
            Console.WriteLine(y);


        }
    }
}
