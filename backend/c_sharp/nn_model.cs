using System;

namespace C_Sharp_Backend
{
	class Template_Class
    {
        /* Defines the number of layers. Including input and output layer. */
        const uint NUMBER_OF_LAYERS = ###numberLayers###;

        /* Defines the width of the output of each layer when seen as a matrix. */
        const uint LAYER_OUTPUT_WIDTH[###numberLayers###] = ###layerOutputWidth###;

        /* Defines the height of the output of each layer when seen as a matrix. */
        const uint LAYER_OUTPUT_HEIGHT[###numberLayers###] = ###layerOutputHeight###;

        /* Defines the depth of the output of each layer when seen as a matrix. */
        const uint LAYER_OUTPUT_DEPTH[###numberLayers###] = ###layerOutputDepth###;

        /* Defines the type of the layer. See layer_type enumeration for possible values. */
        const uint LAYER_TYPE[###dimNumberLayers###] = ###layerTypes###;

        /* Defines the type of activation function for the layer. Default value for layers without activation function is 0. */
        const uint ACTIVATION_FUNCTION[###dimNumberLayers###] = ###activationFunctions###;

        /*  Defines the index at which the first weight-element of each layer is present. */
        const uint WEIGHTS_START_INDEX[###dimNumberLayers###] = ###indicesWeights###;

        /* Defines the index at which the first bias-element of each layer is present. */
        const uint BIASES_START_INDEX[###dimNumberLayers###] = ###indicesBias###;

        /* Defines whether bias values should be applied to the layer. */
        const uint BIAS_ENABLED[###dimNumberLayers###] = ###useBias###;

        /* Holds the weights for each layer as flatted one-dimensional array. */
        const float WEIGHTS[###dimWeights###] = ###weights###;

        /* Holds the biases for each layer as flatted one-dimensional array. */
        const float BIASES[###dimBias###] = ###bias###;

        /* Defines the width of the filter/pool for each layer when seen as a matrix.
        Default value for layers without filters/pools is 0. */
        const uint POOL_WIDTH[###dimNumberLayers###] = ###poolWidth###;

        /* Defines the height of the filter/pool for each layer when seen as a matrix.
        Default value for layers without filters/pools is 0. */
        const uint POOL_HEIGHT[###dimNumberLayers###] = ###poolHeight###;

        /* Defines the horizontal stride/step size for the filter/pool for each layer.
        Default value for layers without filters/pools is 0. */
        const uint HORIZONTAL_STRIDE[###dimNumberLayers###] = ###horizontalStride###;

        /* Defines the vertical stride/step size for the filter/pool for each layer.
        Default value for layers without filters/pools is 0. */
        const uint VERTICAL_STRIDE[###dimNumberLayers###] = ###verticalStride###;

        /* Defines if padding should be applied for each layer. See padding enumeration for possible values. */
        const uint PADDING[###dimNumberLayers###] = ###padding###;

        /* Performs the prediction for a given set of input values. The input lenght must match the specification. */
        // float * predict(float * input);

        /* Functions for each layer are specified in .h file to allow direct references for testing purposes.
        Otherwise only prediction should be referenced externally.
        The name of each helper function is composed as follows: LAYERNAME_(IF AVAILABLE:TYPE)_ACTION */
        static float* activation_apply(float* input, uint16_t input_columns, uint16_t input_rows, uint16_t input_depth, uint8_t activation);
        static float* bias_apply(float* input, uint16_t input_length, const float biases[], uint16_t bias_start_index);
        static float* dense_apply(float* input, uint16_t number_of_previous_units, uint16_t number_of_current_units, const float weights[], uint16_t weights_start_index, const float biases[], uint16_t bias_start_index, uint8_t use_bias, uint8_t activation);
        static float* padding_zero_apply(float* input, uint16_t input_columns, uint16_t input_rows, uint16_t input_depth, uint16_t pool_size_width, uint16_t pool_size_height);
        static float* pooling_apply(float* input, uint16_t input_columns, uint16_t input_rows, uint16_t input_depth, uint8_t pooling_type, uint16_t pool_size_width, uint16_t pool_size_height, uint16_t horizontal_stride, uint16_t vertical_stride, uint8_t padding, uint16_t output_columns, uint16_t output_rows);
        static float* pooling_avg_apply(float* input, uint16_t input_columns, uint16_t input_rows, uint16_t input_depth, uint16_t pool_size_width, uint16_t pool_size_height, uint16_t horizontal_stride, uint16_t vertical_stride, uint16_t output_columns, uint16_t output_rows);
        static float* pooling_max_apply(float* input, uint16_t input_columns, uint16_t input_rows, uint16_t input_depth, uint16_t pool_size_width, uint16_t pool_size_height, uint16_t horizontal_stride, uint16_t vertical_stride, uint16_t output_columns, uint16_t output_rows);
        static float* padding_values_apply(float* input, uint16_t input_columns, uint16_t input_rows, uint16_t input_depth, uint16_t number_of_padding_layers);

        /* Helper functions to perform calculations*/
        static uint16_t padding_calculate_size(uint16_t pool_size);
        static uint16_t padding_calculate_output_size(uint16_t input_size, uint16_t padding_size);
        static float activation_function_apply(uint8_t activation, float value, float denominator);

        /*
        Purpose: Generates the output predictions for the input samples.
        Arguments:
        - input: (Flattened) input values as array
        Returns: (Flattened) output array
        */
        static float[] predict(float[] input)
        {
            uint current_layer_index;

            /* Loops through each layer of the neural network.
            The initial value is set to 1, since the layer at index 0 is the input layer
            and there is no transformation required at the input layer level. */
            for (current_layer_index = 1; current_layer_index < NUMBER_OF_LAYERS; current_layer_index++)
            {
                //Dense
                if (LAYER_TYPE[current_layer_index - 1] == lt_dense)
                {
                    uint8_t activation = ACTIVATION_FUNCTION[current_layer_index - 1];
                    uint16_t number_of_previous_units = LAYER_OUTPUT_HEIGHT[current_layer_index - 1];
                    uint16_t number_of_current_units = LAYER_OUTPUT_HEIGHT[current_layer_index];
                    uint8_t use_bias = BIAS_ENABLED[current_layer_index - 1];
                    uint16_t bias_start_index = BIASES_START_INDEX[current_layer_index - 1];
                    uint16_t weights_start_index = WEIGHTS_START_INDEX[current_layer_index - 1];

                    input = dense_apply(input, number_of_previous_units, number_of_current_units, WEIGHTS, weights_start_index, BIASES, bias_start_index, use_bias, activation);
                }
                //Max and avg pooling
                else if (LAYER_TYPE[current_layer_index - 1] == lt_max_pooling || LAYER_TYPE[current_layer_index - 1] == lt_avg_pooling)
                {
                    uint16_t input_columns = LAYER_OUTPUT_WIDTH[current_layer_index - 1];
                    uint16_t input_rows = LAYER_OUTPUT_HEIGHT[current_layer_index - 1];
                    uint16_t input_depth = LAYER_OUTPUT_DEPTH[current_layer_index - 1];

                    uint16_t output_columns = LAYER_OUTPUT_WIDTH[current_layer_index];
                    uint16_t output_rows = LAYER_OUTPUT_HEIGHT[current_layer_index];
                    uint16_t output_depth = LAYER_OUTPUT_DEPTH[current_layer_index];

                    uint16_t pool_size_width = POOL_WIDTH[current_layer_index];
                    uint16_t pool_size_height = POOL_HEIGHT[current_layer_index];
                    uint16_t horizontal_stride = HORIZONTAL_STRIDE[current_layer_index];
                    uint16_t vertical_stride = VERTICAL_STRIDE[current_layer_index];
                    uint8_t padding = PADDING[current_layer_index];
                    uint8_t pooling_type = LAYER_TYPE[current_layer_index - 1];
                    input = pooling_apply(input, input_columns, input_rows, input_depth, pooling_type, pool_size_width, pool_size_height, horizontal_stride, vertical_stride, padding, output_columns, output_rows);
                }
                //Activation
                else if (LAYER_TYPE[current_layer_index - 1] == lt_activation)
                {
                    uint16_t input_columns = LAYER_OUTPUT_WIDTH[current_layer_index - 1];
                    uint16_t input_rows = LAYER_OUTPUT_HEIGHT[current_layer_index - 1];
                    uint16_t input_depth = LAYER_OUTPUT_DEPTH[current_layer_index - 1];
                    uint8_t activation = ACTIVATION_FUNCTION[current_layer_index - 1];
                    input = activation_apply(input, input_columns, input_rows, input_depth, activation);
                }

                /* We do not need to do anything for flatten layers since we're already using a flattened array structure
                  and the calculation of the proper sizing is already performed by the python backend. */
            }

            return input;
        }


    }
}