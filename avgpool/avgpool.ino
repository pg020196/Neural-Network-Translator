void setup()
{
  Serial.begin(9600);
  Serial.println();
  test_for_zero_padding();
  Serial.println();
  test_for_avg_pooling();
  Serial.println();
  test_for_max_pooling();
}

void loop()
{

}

void test_for_max_pooling() {
  int input_width = 6;
  int input_height = 1;
  int pool_width = 2;
  int pool_height  = 1;
  int horizontal_stride = 1;
  int vertical_stride = 1;
  float input[input_width * input_height] = {1, 2, 3, 2, 1, 5};

  apply_max_pool(input, input_width, input_height, pool_width, pool_height, horizontal_stride, vertical_stride);
}


void test_for_avg_pooling() {
  int input_width = 6;
  int input_height = 1;
  int pool_width = 2;
  int pool_height  = 1;
  int horizontal_stride = 1;
  int vertical_stride = 1;
  float input[input_width * input_height] = {1, 2, 3, 4, 5, 6};

  apply_avg_pool(input, input_width, input_height, pool_width, pool_height, horizontal_stride, vertical_stride, 0);
}

void test_for_zero_padding()
{
  int input_width = 6;
  int input_height = 4;
  int pool_width = 3;
  int pool_height = 3;
  float input[input_width * input_height] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4};

  apply_zero_padding(input, input_width, input_height, pool_height, pool_width);
}

void apply_zero_padding(float input[], int input_width, int input_height, int pool_width, int pool_height)
{
  int current_row;
  int current_column;
  int output_index;
  int input_index = 0;
  int vertical_padding = (pool_height - 1) / 2;
  int horizontal_padding = (pool_width - 1) / 2;
  int output_width = input_width + horizontal_padding * 2;
  int output_height = input_height + vertical_padding * 2;
  float output[output_width * output_height];
  memset(output, 0, output_width * output_height * sizeof(float));

  for (current_row = 0; current_row < output_height; current_row++)
  {
    for (current_column = 0; current_column < output_width; current_column++)
    {
      if (current_row >= vertical_padding && current_row < (output_height - vertical_padding) && current_column >= horizontal_padding && current_column < (output_width - horizontal_padding))
      {
        output_index = current_row * output_width + current_column;
        output[output_index] = input[input_index];
        input_index++;
      }
    }
  }
  print_matrix(output, output_width, output_height);
}

/*
   Padding == 1 is activated
*/
void apply_avg_pool(float input[], int input_width, int input_height, int pool_width, int pool_height, int horizontal_stride, int vertical_stride, int padding)
{
  int vertical_padding = 0;
  int horizontal_padding = 0;

  int output_height = ((input_height - pool_height + 2 * vertical_padding) / vertical_stride) + 1;
  int output_width = ((input_width - pool_width + 2 * horizontal_padding) / horizontal_stride) + 1;

  Serial.println(String(output_width) + "," + String(output_height));

  float output[output_height * output_width];
  memset(output, 0, output_height * output_width * sizeof(float));

  for (int vertical_step_index = 0; vertical_step_index < output_height ; vertical_step_index += vertical_stride)
  {
    for (int horizontal_step_index = 0; horizontal_step_index < output_width; horizontal_step_index += horizontal_stride)
    {
      float result = 0;
      for (int vertical_pool_index = 0; vertical_pool_index < pool_height; vertical_pool_index++)
      {
        for (int horizontal_pool_index = 0; horizontal_pool_index < pool_width; horizontal_pool_index++)
        {
          int index = horizontal_pool_index + vertical_pool_index * input_width + vertical_step_index * input_width + horizontal_step_index;
          result = result + input[index];
        }
      }
      output[horizontal_step_index + vertical_step_index * pool_width] = float(result) / (pool_width * pool_height);
    }
  }
  print_matrix(output, output_width, output_height);
}

void apply_max_pool(float input[], int input_width, int input_height, int pool_width, int pool_height, int horizontal_stride, int vertical_stride)
{
  int vertical_padding = 0;
  int horizontal_padding = 0;

  int output_height = ((input_height - pool_height + 2 * vertical_padding) / vertical_stride) + 1;
  int output_width = ((input_width - pool_width + 2 * horizontal_padding) / horizontal_stride) + 1;

  Serial.println(String(output_width) + "," + String(output_height));

  float output[output_height * output_width];
  memset(output, 0, output_height * output_width * sizeof(float));

  for (int vertical_step_index = 0; vertical_step_index < output_height ; vertical_step_index += vertical_stride)
  {
    for (int horizontal_step_index = 0; horizontal_step_index < output_width; horizontal_step_index += horizontal_stride)
    {
      float result = input[vertical_step_index * input_width + horizontal_step_index];
      for (int vertical_pool_index = 0; vertical_pool_index < pool_height; vertical_pool_index++)
      {
        for (int horizontal_pool_index = 1; horizontal_pool_index < pool_width; horizontal_pool_index++)
        {
          int index = horizontal_pool_index + vertical_pool_index * input_width + vertical_step_index * input_width + horizontal_step_index;
          if (input[index] > result)
          {
            result = input[index];
          }
        }
      }
      output[horizontal_step_index + vertical_step_index * pool_width] = result;
    }
  }
  print_matrix(output, output_width, output_height);
}

void print_matrix(float matrix[], int width, int height)
{
  Serial.println();
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      Serial.print(String(matrix[i * width + j]) + "\t");
    }
    Serial.println();
  }
}
