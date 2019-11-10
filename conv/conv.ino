void setup() {

  /* Opens the serial port and sets the data rate to 9600 bps. */
  Serial.begin(9600);
  //test_padding();
  Serial.println();
  test_convolution();

}

void test_padding()
{
  //Input variables
  uint16_t number_of_padding_layers = 1;
  uint16_t input_width = 1;
  uint16_t input_height = 4;
  uint16_t output_width = input_width + 2 * number_of_padding_layers;
  uint16_t output_height = input_height + 2 * number_of_padding_layers;

  float * input = calloc(input_width * input_height, sizeof(float));
  float data[input_width * input_height] = {1, 2, 3, 4};
  for (uint16_t i = 0; i < input_width * input_height; i++)
  {
    *(input + i) = data[i];
  }

  float * result = padding_values_apply(input, input_width, input_height, number_of_padding_layers);

  Serial.println();
  for (uint16_t i = 0; i < output_height; i++) {
    for (uint16_t j = 0; j < output_width; j++) {
      uint16_t print_index = i * output_width + j;
      Serial.print(String(*(result + print_index)) + "  ");
    }
    Serial.println();
  }
}

void test_convolution()
{
  uint16_t input_width = 6;
  uint16_t input_height = 6;
  float * input = calloc(input_width * input_height, sizeof(float));
  float data[input_width * input_height] = {7, 6, 5, 5, 6, 7, 6, 4, 3, 3, 4, 6, 5, 3, 2, 2, 3, 5, 5, 3, 2, 2, 3, 5, 6, 4, 3, 3, 4, 6, 7, 6, 5, 5, 6, 7};
  for (uint16_t i = 0; i < input_width * input_height; i++)
  {
    *(input + i) = data[i];
  }

  
  uint16_t kernel_width = 3;
  uint16_t kernel_height = 3;
  const float kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
  uint16_t vertical_stride = 1;
  uint16_t horizontal_stride = 1;
  uint16_t output_height = 6;
  uint16_t output_width = 6;

  float * output = convolution_apply(input, input_width, input_height, kernel, kernel_width, kernel_height, horizontal_stride, vertical_stride, output_width, output_height);

  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      int print_index = i * output_width + j;
      Serial.print(String(*(output + print_index)) + "  ");
    }
    Serial.println();
  }
}

float * convolution_apply(float * input, uint16_t input_width, uint16_t input_height, const float kernel[], uint16_t kernel_width, uint16_t kernel_height, uint16_t horizontal_stride, uint16_t vertical_stride, uint16_t output_width, uint16_t output_height)
{
  uint16_t output_row_index;
  uint16_t output_column_index;
  uint16_t kernel_row_index;
  uint16_t kernel_column_index;
  uint16_t input_index;
  uint16_t kernel_index;
  uint16_t output_index;
  float result;
  float * output;

  uint16_t number_of_padding_layers = 1;

  input = padding_values_apply(input, input_width, input_height, number_of_padding_layers);

  input_width = input_width + number_of_padding_layers * 2;
  input_height = input_height + number_of_padding_layers * 2;

  output = calloc(output_height * output_width, sizeof(float));

  //Loop through output matrix
  for (output_row_index = 0; output_row_index < output_height; output_row_index += vertical_stride)
  {
    //Loop through output matrix
    for (output_column_index = 0; output_column_index < output_width; output_column_index += horizontal_stride)
    {
      result = 0;
      for (kernel_row_index = 0; kernel_row_index < kernel_height; kernel_row_index++)
      {
        for (kernel_column_index = 0; kernel_column_index < kernel_width; kernel_column_index++)
        {
          input_index = kernel_column_index + kernel_row_index * input_width + output_row_index * input_width + output_column_index;
          kernel_index = kernel_row_index * kernel_width + kernel_column_index;
          result = result + (*(input + input_index) * kernel[kernel_index]);
        }
      }
      output_index = output_column_index + output_row_index * output_width;
      *(output + output_index) = result;
    }
  }

  free(input);
  return output;
}

float * padding_values_apply(float * input, uint16_t input_width, uint16_t input_height, uint16_t number_of_padding_layers)
{
  uint16_t current_padding_layer_index;
  uint16_t output_width = input_width;
  uint16_t output_height = input_height;
  int16_t input_index;
  int16_t output_index;
  float * output;
  uint16_t output_column_index;
  uint16_t output_row_index;

  for (current_padding_layer_index = 0; current_padding_layer_index < number_of_padding_layers; current_padding_layer_index++)
  {
    input_index = 0;
    output_width = output_width + 2;
    output_height = output_height + 2;
    output = calloc(output_width * output_height, sizeof(float));

    //Copy values from input to output
    for (output_column_index = 0; output_column_index < output_height; output_column_index++)
    {
      for (output_row_index = 0; output_row_index < output_width; output_row_index++)
      {
        if (!(output_row_index < 1 || output_row_index > input_width || output_column_index < 1 || output_column_index > input_height))
        {
          output_index = output_column_index * output_width + output_row_index;
          output[output_index] = input[input_index];
          input_index = input_index + 1;
        }
      }
    }

    // Copy upper row and extend corners
    for (output_column_index = 0; output_column_index < output_width; output_column_index++)
    {
      input_index = output_column_index - 1;

      if (input_index < 0) {
        input_index = 0;
      }

      if (input_index >= input_width)
      {
        input_index = input_width - 1 ;
      }

      output[output_column_index] = input[input_index];
    }

    // Copy lower row and extend corners
    for (output_column_index = 0; output_column_index < output_width; output_column_index++)
    {
      output_index = output_height * output_width - output_width + output_column_index;
      input_index = input_height * input_width - input_width + output_column_index - 1;

      if (input_index < input_height * input_width - input_width || input_index < 0)
      {
        input_index = input_height * input_width - input_width;
      }

      if (input_index >= input_height * input_width) {
        input_index = input_height * input_width - 1;
      }

      output[output_index] = input[input_index];
    }

    //Extend left side
    for (output_row_index = 1; output_row_index < output_height - 1; output_row_index++)
    {
      output_index = output_row_index * output_width;
      input_index = (output_row_index - 1) * input_width;

      if (input_index < 0)
      {
        input_index = 0;
      }

      output[output_index] = input[input_index];
    }

    // Extend right side
    for (output_row_index = 1; output_row_index < output_height - 1; output_row_index++)
    {
      output_index = output_row_index * output_width + output_width - 1;
      input_index = (output_row_index - 1) * input_width + input_width - 1;

      output[output_index] = input[input_index];
    }

    input_width = output_width;
    input_height = output_height;
    free(input);
    input = output;
  }
  return input;
}

void loop() {

}
