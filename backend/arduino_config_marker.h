const byte numberOfLayers = ###numberLayers###;

/* defines the number of layers between 0 to 255 */
unsigned const int unitsInLayers[###numberLayers###] = ###unitsInLayers###;;

/*
      Defines the layer types
*/
const byte layerType[###dimNumberLayers###] = ###layerTypes###;

/*
   Defines the activation functions for each layer
   0: Linera
   1: Sigmoid
   2: Relu
   3: TanH
   4: Softmax
*/
const byte activationFunctions[###dimNumberLayers###] = ###activationFunctions###;

/*
   Defines the index in which the first weight-element of the layer is present
*/
unsigned const int startIndicesWeights[###dimNumberLayers###] = ###indicesWeights###;

/*
  Defines the index in which the first weight-element of the layer is present
*/
unsigned const int startIndicesBias[###dimNumberLayers###] = ###indicesBias###;

/*
   Defines whether bias values should be applied to the layer
*/
const bool useBias[###dimNumberLayers###] = ###useBias###;

/*
   Holds the weights for each layer as flatted one-dimensional array
*/
const float weights[###dimWeights###] = ###weights###;

/*
   Holds the biases for each layer as flatted one-dimensional array
*/
const float biases[###dimBias###] = ###bias###;
