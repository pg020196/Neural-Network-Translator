extern "C"
{
#include "nn_model.h"
}

void setup() {

  /* Opens the serial port and sets the data rate to 9600 bps. */
  Serial.begin(9600);


  addpadding();
}

void conv()
{
  Serial.println("hi");
  // TODO Implement convolution layer logic
  float input[64] = {7, 7, 6, 5, 5, 6, 7, 7, 7, 7, 6, 5, 5, 6, 7, 7, 6, 6, 4, 3, 3, 4, 6, 6, 5, 5, 3, 2, 2, 3, 5, 5, 5, 5, 3, 2, 2, 3, 5, 5, 6, 6, 4, 3, 3, 4, 6, 6, 7, 7, 6, 5, 5, 6, 7, 7, 7, 7, 6, 5, 5, 6, 7, 7};
  uint16_t input_width = 8;
  uint16_t input_height = 8;
  uint16_t kernel_width = 3;
  uint16_t kernel_height = 3;
  //float * kernel = calloc(kernel_width*kernel_height, sizeof(float));
  float kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
  uint16_t vertical_stride = 1;
  uint16_t horizontal_stride = 1;
  uint16_t vertical_step_index;
  uint16_t horizontal_step_index;
  uint16_t vertical_kernel_index;
  uint16_t horizontal_kernel_index;
  uint16_t input_index;
  uint16_t kernel_index;
  uint16_t output_height = 6;
  uint16_t output_width = 6;
  float result;
  float * output = calloc(output_height * output_width, sizeof(float));
  Serial.println("hi");
  //Loop through output matrix
  for (vertical_step_index = 0; vertical_step_index < output_height; vertical_step_index += vertical_stride)
  {
    //Loop through output matrix
    for (horizontal_step_index = 0; horizontal_step_index < output_width; horizontal_step_index += horizontal_stride)
    {
      result = 0;
      for (vertical_kernel_index = 0; vertical_kernel_index < kernel_height; vertical_kernel_index++)
      {
        for (horizontal_kernel_index = 0; horizontal_kernel_index < kernel_width; horizontal_kernel_index++)
        {
          input_index = horizontal_kernel_index + vertical_kernel_index * input_width + vertical_step_index * input_width + horizontal_step_index;
          kernel_index = vertical_kernel_index * kernel_width + horizontal_kernel_index;
          result = result + (*(input + input_index) * kernel[kernel_index]);
        }
      }
      Serial.print(String(result) + ",");
      *(output + (horizontal_step_index + vertical_step_index * kernel_width)) = result;
    }
    Serial.println();
  }

  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      int print_index = i * output_width + j;
      Serial.print(String(*(output + print_index)) + "  ");
    }
    Serial.println();
  }
}

void addpadding()
{
  //Input variables
  int number_of_padding_layers = 1;

  //Local varibles
  int currend_padding_layer;
  for(current_padding_layer = 0; current_padding_layer < number_of_padding_layers; current_padding_layer++)
  {
    Serial.println();
  int input_width = 6;
  int input_height = 6;
  int output_width = 8;
  int output_height = 8;

  float input[input_width * input_height] = {7,6,5,5,6,7,6,4,3,3,4,6,5,3,2,2,3,5,5,3,2,2,3,5,6,4,3,3,4,6,7,6,5,5,6,7};
  float output[output_width * output_height];
  
  int counter = 0;
  memset(output, 0, output_width * output_height * sizeof(float));

  //Copy values from input to output
  for (int o_h_index = 0; o_h_index < output_height; o_h_index++)
  {
    for (int o_w_index = 0; o_w_index < output_width; o_w_index++)
    {
      if(!(o_w_index < 1 || o_w_index > input_width || o_h_index < 1 || o_h_index > input_height))
      {
        int output_index = o_h_index*output_width + o_w_index;
        output[output_index] = input[counter];
        counter = counter + 1;
      }
    }
  }
  
  // Copy upper row and extend corners
  for(int i = 0; i < output_width; i++)
  {
    
    int input_index = i-1;
    if(input_index < 0){
      input_index = 0;
    }
    if(input_index >= input_width)
    {
      input_index = input_width - 1 ;
    }
    output[i] = input[input_index];
  }

  // Copy lower row and extend corners
  for(int i = 0; i < output_width; i++)
  {
   int output_index = output_height*output_width-output_width+i;
   Serial.print(String(output_index)+",");

   int input_index = input_height*input_width-input_width+i-1;

   if(input_index < input_height*input_width-input_width)
   {
    input_index = input_height*input_width-input_width;
   }

   if(input_index >= input_height*input_width){
    input_index = input_height*input_width - 1;
   }
   
   Serial.print(String(input_index)+",");
   output[output_index] = input[input_index]; 

   Serial.println();
  }

  //Extend left side
  for(int i = 1; i < output_height - 1; i++)
  {
    int output_index = i*output_height;
    int input_index = (i-1)*input_width;
    if(input_index < 0)
    {
      input_index = 0;
    }
    output[output_index] = input[input_index]; 
  }

  // Extend right side
  for(int i = 1; i < output_height - 1; i++)
  {
    int output_index = i*output_height+output_width-1;
    int input_index = (i-1)*input_width+input_width-1;
    
    output[output_index] = input[input_index]; 
  }

  Serial.println();
  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      int print_index = i * output_width + j;
      Serial.print(String(output[print_index]) + "  ");
    }
    Serial.println();
  }
  }
  

}

void loop() {

}
