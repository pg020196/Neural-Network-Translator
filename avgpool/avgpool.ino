

void setup()
{
  Serial.begin(9600);
  Serial.println();
  maxpool2d();
  Serial.println();
  avgpool2d();
  

}
void loop()
{

}

void avgpool2d()
{
    int input_matrix[25] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  int matrix_width = 5;
  int matrix_height = 5;
  int pool_width = 3;
  int pool_height  = 3;
  int step_width = 1;
  int step_height = 1;
  float result_matrix[pool_height * pool_width] = { 0 };


  for (int i = 0; i < pool_height; i += step_height)
  {
    for (int j = 0; j < pool_width; j += step_width) {

      float result = 0;
      for (int k = 0; k < pool_height; k++)
      {
        for (int l = 0; l < pool_width; l++)
        {
          int index = l + k * matrix_width + i * matrix_width + j;
          result += input_matrix[index];
        }
      }
      result_matrix[j + i * pool_height] = float(result) / (pool_width * pool_height);
    }

  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Serial.print(String(result_matrix[j + i * 3]) + "     ");
    }
    Serial.println();
  }

}

void maxpool2d()
{
  int input_matrix[25] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  int matrix_width = 5;
  int matrix_height = 5;
  int pool_width = 3;
  int pool_height  = 3;
  int step_width = 1;
  int step_height = 1;
  float result_matrix[pool_height * pool_width] = { 0 };


  for (int i = 0; i < pool_height; i += step_height)
  {
    for (int j = 0; j < pool_width; j += step_width) {
      float result = input_matrix[i * matrix_width + j];
      for (int k = 0; k < pool_height; k++)
      {
        // Init with 1 since the result value has 
        for (int l = 1; l < pool_width; l++)
        {
          int index = l + k * matrix_width + i * matrix_width + j;
          if(input_matrix[index] > result)
          {
            result = input_matrix[index];
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
