extern "C"
{
#include "nn_model.h"
}

#include <stdlib.h>

void setup() {

  /* Opens the serial port and sets the data rate to 9600 bps. */
  Serial.begin(9600);
  Serial.println("Enter input values:");

  /* Get the number of bytes (characters) available for reading from the serial port.
  This is data that’s already arrived and stored in the serial receive buffer (which holds 64 bytes). */


    /* Inits the input float array with zeros.
    The length of the array is determined through the number of units in layer 0 (input layer). */
    float * input = calloc(LAYER_OUTPUT_WIDTH[0]*LAYER_OUTPUT_HEIGHT[0], sizeof(float));
    

    /* Call the predict function with the input array. */
    uint32_t startTime = micros();
    float tmpData[15] = {26,1,37,115,189,31.3,31.3,29.6,103.,0.205,0,83,41,36,2.2};
    for(int i = 0; i < 15; i++){
      *(input +i)=tmpData[i];
    }

    uint16_t input_width = LAYER_OUTPUT_WIDTH[0];
    uint16_t input_height = LAYER_OUTPUT_HEIGHT[0];
    uint16_t input_depth = LAYER_OUTPUT_DEPTH[0];

    uint16_t pool_width = POOL_WIDTH[1];
    uint16_t pool_height = POOL_HEIGHT[1];
    
    //float * result = padding_zero_apply(input, input_width, input_height, input_depth, pool_width, pool_height);
    float * result = predict(input);
    uint32_t endTime = micros();

    
    for(int i = 0; i < LAYER_OUTPUT_HEIGHT[NUMBER_OF_LAYERS-1]; i++)
    {
      for(int j = 0; j < LAYER_OUTPUT_WIDTH[NUMBER_OF_LAYERS-1]; j++)
      {
        //Change to print and add whitespaces
        Serial.print(String(*(result+i*LAYER_OUTPUT_WIDTH[NUMBER_OF_LAYERS-1]+j),8)+"   ");
      }
      Serial.println();
      //Add println
    }

    Serial.println(String(endTime-startTime)+" microseconds");
    Serial.println("\nEnter input values:");
  
}

void loop()
{
  
}



static float *pooling_max_apply(float *input, uint16_t input_width, uint16_t input_height, uint16_t pool_width, uint16_t pool_height, uint16_t horizontal_stride, uint16_t vertical_stride, uint16_t output_width, uint16_t output_height, uint16_t output_depth)
{
  uint16_t current_z_index;
  uint16_t vertical_step_index;
  uint16_t horizontal_step_index;
  uint16_t vertical_pool_index;
  uint16_t horizontal_pool_index;
  uint16_t input_index;
  uint16_t output_index = 0;
  float result;
  uint16_t initial_result_index;
  float *output = calloc(output_height * output_width * output_height, sizeof(float));

  for (current_z_index = 0; current_z_index < output_depth; current_z_index++)
  {
    for (vertical_step_index = 0; vertical_step_index <= input_height - pool_height; vertical_step_index += vertical_stride)
    {
      for (horizontal_step_index = 0; horizontal_step_index <= input_width - pool_width; horizontal_step_index += horizontal_stride)
      {
        //Obere linke Ecke der Filtermatrix
        initial_result_index = (vertical_step_index * input_width + horizontal_step_index) + current_z_index * (input_width * input_height);
        //Serial.println(String(initial_result_index));
        result = *(input + initial_result_index);
        
        for (vertical_pool_index = 0; vertical_pool_index < pool_height; vertical_pool_index++)
        {
          for (horizontal_pool_index = 0; horizontal_pool_index < pool_width; horizontal_pool_index++)
          {
            input_index = initial_result_index + vertical_pool_index * input_width + horizontal_pool_index;
            Serial.println(input_index);
            if (*(input + input_index) > result)
            {
              result = *(input + input_index);
            }
          }
          
        }
        Serial.println("-");
        *(output + output_index) = result;
        output_index++;
      }
    }
  }
  free(input);
  return output;
}

static float *pooling_avg_apply(float *input, uint16_t input_width, uint16_t input_height, uint16_t pool_width, uint16_t pool_height, uint16_t horizontal_stride, uint16_t vertical_stride, uint16_t output_width, uint16_t output_height, uint16_t output_depth)
{
  uint16_t current_z_index;
  uint16_t vertical_step_index;
  uint16_t horizontal_step_index;
  uint16_t vertical_pool_index;
  uint16_t horizontal_pool_index;
  uint16_t input_index;
  uint16_t output_index = 0;
  float result;
  uint16_t initial_result_index;
  float *output = calloc(output_height * output_width * output_height, sizeof(float));

  for (current_z_index = 0; current_z_index < output_depth; current_z_index++)
  {
    for (vertical_step_index = 0; vertical_step_index <= input_height - pool_height; vertical_step_index += vertical_stride)
    {
      for (horizontal_step_index = 0; horizontal_step_index <= input_width - pool_width; horizontal_step_index += horizontal_stride)
      {
        //Obere linke Ecke der Filtermatrix
        initial_result_index = (vertical_step_index * input_width + horizontal_step_index) + current_z_index * (input_width * input_height);
        //Serial.println(String(initial_result_index));
        result = 0;
        
        for (vertical_pool_index = 0; vertical_pool_index < pool_height; vertical_pool_index++)
        {
          for (horizontal_pool_index = 0; horizontal_pool_index < pool_width; horizontal_pool_index++)
          {
            input_index = initial_result_index + vertical_pool_index * input_width + horizontal_pool_index;
            result = result + *(input+input_index);
          }
          
        }
        
        *(output + output_index) = (float)result / (pool_width*pool_height);
        output_index++;
      }
    }
  }
  free(input);
  return output;
}

float * pooling_apply(uint8_t pooling_type, float *input, uint16_t input_width, uint16_t input_height, uint16_t input_depth, uint16_t output_width, uint16_t output_height, uint16_t output_depth, uint16_t pool_width, uint16_t pool_height, uint16_t horizontal_stride, uint16_t vertical_stride, uint8_t padding)
{
  if (pooling_type == 3)
  {
    //return pooling_max_apply(input, input_width, input_height, pool_width, pool_height, horizontal_stride, vertical_stride, output_width, output_height, output_depth);
    return pooling_avg_apply(input, input_width, input_height, pool_width, pool_height, horizontal_stride, vertical_stride, output_width, output_height, output_depth);
  }
  else
  {
    return input;
  }
}

float *predict(float *input)
{
  uint16_t current_layer_index;

  /* Loops through each layer of the neural network.
  The initial value is set to 1, since the layer at index 0 is the input layer
  and there is no transformation required at the input layer level. */
  for (current_layer_index = 1; current_layer_index < NUMBER_OF_LAYERS; current_layer_index++)
  {
    //Max and avg pooling
    if (LAYER_TYPE[current_layer_index - 1] == 3 || LAYER_TYPE[current_layer_index - 1] == 4)
    {
      uint16_t input_width = LAYER_OUTPUT_WIDTH[current_layer_index - 1];
      uint16_t input_height = LAYER_OUTPUT_HEIGHT[current_layer_index - 1];
      uint16_t input_depth = LAYER_OUTPUT_DEPTH[current_layer_index - 1];

      uint16_t output_width = LAYER_OUTPUT_WIDTH[current_layer_index];
      uint16_t output_height = LAYER_OUTPUT_HEIGHT[current_layer_index];
      uint16_t output_depth = LAYER_OUTPUT_DEPTH[current_layer_index];

      uint16_t pool_width = POOL_WIDTH[current_layer_index];
      uint16_t pool_height = POOL_HEIGHT[current_layer_index];
      uint16_t horizontal_stride = HORIZONTAL_STRIDE[current_layer_index];
      uint16_t vertical_stride = VERTICAL_STRIDE[current_layer_index];
      uint8_t padding = PADDING[current_layer_index];

      input = pooling_apply(LAYER_TYPE[current_layer_index - 1], input, input_width, input_height, input_depth, output_width, output_height, output_depth, pool_width, pool_height, horizontal_stride, vertical_stride, padding);
    }
  }

  /* We do not need to do anything for flatten layers since we're already using a flattened array structure
      and the calculation of the proper sizing is already performed by the python backend. */

  return input;
}

static float *padding_zero_apply(float *input, uint16_t input_width, uint16_t input_height, uint16_t input_depth, uint16_t pool_width, uint16_t pool_height)
{
  uint16_t current_x_index;
  uint16_t current_y_index;
  uint16_t current_z_index;
  uint16_t output_index;
  uint16_t input_index;

  uint16_t vertical_padding = padding_calculate_size(pool_height);
  uint16_t horizontal_padding = padding_calculate_size(pool_width);

  //! Create own arguments for these variables and move this method call to the predict method
  uint16_t output_width = padding_calculate_output_size(input_width, horizontal_padding);
  uint16_t output_height = padding_calculate_output_size(input_height, vertical_padding);
  Serial.println(output_width);
  uint16_t output_depth = input_depth;

  float *output = calloc(output_width * output_height * output_depth, sizeof(float));

  for (current_z_index = 0; current_z_index < output_depth; current_z_index++)
  {
    input_index = current_z_index * output_depth;
    for (current_y_index = 0; current_y_index < output_height; current_y_index++)
    {
      for (current_x_index = 0; current_x_index < output_width; current_x_index++)
      {
        if (current_y_index >= vertical_padding && current_y_index < (output_height - vertical_padding) && current_x_index >= horizontal_padding && current_x_index < (output_width - horizontal_padding))
        {
          output_index = current_y_index * output_width + current_x_index + current_z_index * (output_height * output_width);
          *(output + output_index) = *(input + input_index);
          input_index = input_index + 1;
        }
      }
    }
  }
  free(input);
  return output;
}

static uint16_t padding_calculate_size(uint16_t pool_dimension)
{
  return (pool_dimension - 1) / 2;
}

static uint16_t padding_calculate_output_size(uint16_t input_dimension, uint16_t padding_dimension)
{
  return input_dimension + padding_dimension * 2;
}
