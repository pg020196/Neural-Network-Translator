using System;
using System.Collections.Generic;
using NeuralNetwork;
using Layers;
using Tensor;

namespace NeuralNetworkTemplate
{
	class NeuralNetworkTemplate
    {
        static private ActivationType decodeActivationType(int code)
        {
            switch (code)
            {
                case (0):
                    return ActivationType.linear;
                    break;
                case (1):
                    return ActivationType.sigmoid;
                    break;
                case (2):
                    return ActivationType.relu;
                    break;
                case (3):
                    return ActivationType.tanh;
                    break;
                case (4):
                    return ActivationType.softmax;
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        public static void Main(string[] args)
        {
            int batchSize = 100;
            int numLayers = 2;
            int[] arrayWidth = {1,1,1};
            int[] arrayHeight = {8,8,1};
            int[] arrayDepth = {1,1,1};
            int[] layerTypes = {1,1};
            int[] activationTypes = {1,1};
            var weights = new Tensor<double>(new double[] {0.4021962583065033,0.35737255215644836,-0.4617685079574585,1.1883320808410645,-0.47795209288597107,-0.23023425042629242,0.29660266637802124,0.7326241731643677,1.3849374055862427,-1.380560278892517,-2.417308807373047,0.009452342055737972,-0.09232033789157867,-1.3136069774627686,-0.1561698466539383,2.0229179859161377,0.11960292607545853,0.040415748953819275,0.4152451455593109,-0.6607005000114441,-0.020997274667024612,-0.030194925144314766,0.4268653392791748,0.24414455890655518,0.14898188412189484,0.16087287664413452,0.12000391632318497,1.064346194267273,-0.42680081725120544,0.20197656750679016,0.4484856426715851,-0.9404270052909851,0.8603695631027222,0.18136800825595856,0.3122807443141937,-0.9861078262329102,0.4423156678676605,-0.9267410635948181,0.1835067868232727,-1.1296892166137695,0.8970383405685425,0.06321218609809875,0.3643255829811096,0.8269380331039429,2.3783342838287354,-0.9780294895172119,-0.7461061477661133,-0.06788039952516556,1.782180905342102,0.9131324887275696,-0.40645965933799744,0.3044227957725525,0.5994855165481567,-1.030243992805481,-0.7820641994476318,0.16339054703712463,-0.6115710735321045,0.18630920350551605,-2.594270944595337,-1.3488826751708984,0.04529160261154175,0.4379783868789673,-1.6690258979797363,2.525305986404419,0.47138240933418274,-1.3780920505523682,-1.5163912773132324,1.4166691303253174,0.9666982889175415,-1.393097162246704,-0.7562595009803772,0.9997671246528625});
            int dimWeights = 72;
            int[] indicesWeights = {0,64};
            var bias = new Tensor<double>(new double[] {-0.6492048501968384,0.50642329454422,-1.0451167821884155,-0.49644047021865845,0.5026719570159912,0.5008155703544617,-0.3537411093711853,0.893446683883667,-0.14548198878765106});
            int dimBias = 9;
            int[] indicesBias = {0,8};
            int[] useBias = {1,1};

            if (arrayWidth[0] != 1 || arrayDepth[0] != 1)
                throw new ArgumentException();
            var inputLayer = new InputLayer<double>(new int[] { arrayHeight[0] });

            var layerList = new List<BaseLayer<double>>();
            layerList.Add(inputLayer);

            BaseLayer<double> prevLayer = inputLayer;

            for (int layer_idx = 1; layer_idx <= numLayers; layer_idx++)
            {
                switch (layerTypes[layer_idx - 1])
                {
                    case (1): // dense layer
                        int weightsIndexUpper;
                        int biasIndexUpper;
                        if (layer_idx == numLayers)
                        {
                            weightsIndexUpper = weights.Shape[0];
                            biasIndexUpper = bias.Shape[0];
                        }
                        else
                        {
                            weightsIndexUpper = indicesWeights[layer_idx];
                            biasIndexUpper = indicesBias[layer_idx];
                        }
                        var layer = new Dense<double>(prevLayer.OutputShape, arrayHeight[layer_idx],
                            decodeActivationType(activationTypes[layer_idx - 1]), Convert.ToBoolean(useBias[layer_idx - 1]));

                        int[] weightsShape = { prevLayer.OutputShape[0], arrayHeight[layer_idx] };
                        var layerWeights = weights[indicesWeights[layer_idx - 1]..weightsIndexUpper];
                        layerWeights = layerWeights.reshape(weightsShape);

                        int biasShape = arrayHeight[layer_idx];
                        var layerBias = bias[indicesBias[layer_idx - 1]..biasIndexUpper];
                        layerBias = layerBias.reshape(biasShape);

                        layer.Weights = layerWeights;
                        layer.Bias = layerBias;

                        layerList.Add(layer);
                        prevLayer = layer;
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            var net = new NeuralNetwork<double>(layerList);

            var X = Tensor<double>.randNormal(0, 1, batchSize, arrayHeight[0], arrayWidth[0], arrayDepth[0]);
            if (layerTypes[0] == 1)
                X = X.reshape(batchSize, arrayHeight[0]);

            var y = net.predict(X);
            Console.WriteLine(y);

        }
    }
}