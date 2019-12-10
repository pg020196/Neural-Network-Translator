
/* Defines the number of layers. */
const uint16_t NUMBER_OF_LAYERS = 2;

/* */
//? NEW
const uint16_t LAYER_OUTPUT_WIDTH[2] = {3,2};
const uint16_t LAYER_OUTPUT_HEIGHT[2] = {5,4};

const uint16_t LAYER_OUTPUT_DEPTH[2] = {1,1};

/* Defines the layer types */
const uint8_t LAYER_TYPE[1] = {3};

/*
   Defines the activation functions for each layer as follows:
   0: Linera
   1: Sigmoid
   2: Relu
   3: TanH
   4: Softmax
*/
const uint8_t ACTIVATION_FUNCTION[1] = {0};

/*  Defines the index in which the first weight-element of the layer is present. */
const uint16_t WEIGHTS_START_INDEX[1] = {0};

/* Defines the index in which the first weight-element of the layer is present. */
const uint16_t BIASES_START_INDEX[1] = {0};

/* Defines whether bias values should be applied to the layer. */
const uint8_t BIAS_ENABLED[1] = {0};

/* Holds the weights for each layer as flatted one-dimensional array. */
const float WEIGHTS[0] = {};

/* Holds the biases for each layer as flatted one-dimensional array. */
const float BIASES[0] = {};

//? NEW
const uint16_t POOL_WIDTH[1] = {2};

//? NEW
const uint16_t POOL_HEIGHT[1] = {2};

//? NEW
const uint16_t HORIZONTAL_STRIDE[1] = {1};

//? NEW
const uint16_t VERTICAL_STRIDE[1] = {1};

//? NEW
const uint8_t PADDING[1] = {0};
