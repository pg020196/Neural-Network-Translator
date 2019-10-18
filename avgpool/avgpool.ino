void setup()
{
  Serial.begin(9600);
  Serial.println();
  //maxpool2d(input);
  Serial.println();
  float input[6] = {1,2,3,4,5,6};
  avgpool2d(input);
  
  //apply_zero_padding(input);


}
void loop()
{

}

void apply_zero_padding(float input[])
{
  int pool_height = 2;
  int pool_width = 2;
 
  int vertical_padding = (pool_height - 1) / 2;
  int horizontal_padding = (pool_width - 1) / 2;

  int input_width = 1;
  int input_height = 5;
  
  int output_width = input_width + horizontal_padding * 2;
  int output_height = input_height + vertical_padding * 2;

  float output[output_width * output_height];
  memset(output, 0, output_width*output_height*sizeof(float));

  int input_index = 0;
  int current_row;
  int current_column;
  int output_index;
  
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

  Serial.println();
  for (int i = 0; i < output_width; i++) {
    for (int j = 0; j < output_height; j++) {
      Serial.print(String(output[j + i * output_width]) + "\t");
    }
    Serial.println();
  }

}


void avgpool2d(float input[])
{
  int matrix_width = 6;
  int matrix_height = 1;
  int pool_width = 3;
  int pool_height  = 1;
  int horizontal_stride = 1;
  int vertical_stride = horizontal_stride;

  int vertical_padding = 0;
  int horizontal_padding = 0;

  // https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
  int result_matrix_height = ((matrix_height - pool_height + 2 * vertical_padding) / vertical_stride) + 1;
  int result_matrix_width = ((matrix_width - pool_width + 2 * horizontal_padding) / horizontal_stride) + 1;
  Serial.println(String(result_matrix_width)+","+String(result_matrix_height));
  float result_matrix[result_matrix_height * result_matrix_width] = { 0 };

  // i is the number of vertical steps
  for (int i = 0; i < result_matrix_height ; i += vertical_stride)
  {
    // j is the number of horizontal steps
    for (int j = 0; j < result_matrix_width; j += horizontal_stride) {

      float result = 0;
      for (int k = 0; k < pool_height; k++)
      {
        for (int l = 0; l < pool_width; l++)
        {
          int index = l + k * matrix_width + i * matrix_width + j;
          result += input[index];
        }
      }
      result_matrix[j + i * pool_height] = float(result) / (pool_width * pool_height);
    }

  }

  for (int i = 0; i < result_matrix_height; i++) {
    for (int j = 0; j < result_matrix_width; j++) {
      Serial.print(String(result_matrix[j + i * result_matrix_width]) + "     ");
    }
    Serial.println();
  }

}

void maxpool2d()
{
  int input[25] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  int matrix_width = 5;
  int matrix_height = 5;
  int pool_width = 3;
  int pool_height  = 3;
  int horizontal_stride = 1;
  int vertical_stride = 1;
  float result_matrix[pool_height * pool_width] = { 0 };


  for (int i = 0; i < pool_height; i += vertical_stride)
  {
    for (int j = 0; j < pool_width; j += horizontal_stride) {
      float result = input[i * matrix_width + j];
      for (int k = 0; k < pool_height; k++)
      {
        // Init with 1 since the result value is already initialized with the first entry of the matrix
        for (int l = 1; l < pool_width; l++)
        {
          int index = l + k * matrix_width + i * matrix_width + j;
          if (input[index] > result)
          {
            result = input[index];
          }
        }
      }
      result_matrix[j + i * pool_height] = result;
    }

  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Serial.print(String(result_matrix[j + i * 3]) + "     ");
    }
    Serial.println();
  }


}
