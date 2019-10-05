extern "C" {
#include "backend-config.h"
};

void setup() {

  /*
     Opens the serial port and sets the data rate to 9600 bps
  */
  Serial.begin(9600);
  Serial.println("Enter input values:");
}

void loop()
{
  /*
     Get the number of bytes (characters) available for reading from the serial port. This is data that’s already arrived and stored in the serial receive buffer (which holds 64 bytes).
  */
  if (Serial.available() > 0)
  {
    String inputString;
    /*
       Reads characters from the serial buffer into a String.
    */
    inputString = Serial.readString();

    Serial.println("Input is: " + inputString);

    /*
       Inits the input float array to zeros
       The length of the array is determined through the number of units in layer 0 (input layer)
    */
    float input[unitsInLayers[0]] = {0};
    char *tmp;
    int i = 0;
    /*
       Tokenize the string on "," charakter
    */
    tmp = strtok(&inputString[0], ",");
    while (tmp) {
      /*
         convert a string token to a float
      */
      input[i++] = atof(tmp);
      tmp = strtok(NULL, ",");
    }

    /*
      Call the predict function with the input array
    */
    predict(input);

    Serial.println("\nEnter input values:");
  }
}

void predict(float input[])
{
  /*
       Takes the starting time
  */
  uint32_t startTime = micros();

  /*
     Loops through each layer. Initial layer of current layer = 1 since layer 0 is the input layer.
  */
  for (unsigned int currentLayer = 1; currentLayer < numberOfLayers; currentLayer++)
  {
    unsigned int numberOfPreviousUnits = unitsInLayers[currentLayer - 1];
    unsigned int numberOfCurrentUnits = unitsInLayers[currentLayer];

    /*
       Number of columns in A = Number of Rows in B
    */
    unsigned int p = numberOfPreviousUnits;

    /*
      Number of columns in B
    */
    unsigned int n = numberOfCurrentUnits;

    /*
       Init output values
    */
    float out[n] = {0};

    for (unsigned int i = 0; i < n; i++) {
      if (useBias[currentLayer - 1])
      {
        out[i] = biases[startIndicesBias[currentLayer - 1] + i];
      }
      for (unsigned int j = 0; j < p; j++)
      {
        out[i] = out[i] + input[j] * weights[(n * j + i) + startIndicesWeights[currentLayer - 1]];
      }
      out[i] = apply_activation_function(activationFunctions[currentLayer - 1], out[i]);
    }

    /*
       Re-init the input array with the dimenions of the current number of units
       and set initial value to 0
    */
    memset(input, 0, numberOfCurrentUnits);

    for (unsigned int i = 0; i < numberOfCurrentUnits; i++)
    {
      input[i] = out[i];
    }
  }

  uint32_t endTime = micros();
  Serial.println(input[0], 8);
  Serial.println("done");
  Serial.println(String(endTime - startTime) + " microseconds");
}

/*
   0: Linear
   1: Sigmoid
   2: ReLu
   3: Tanh
   4: Softmax (not implemented)
*/
float apply_activation_function(int actFunc, float input)
{
  switch (actFunc) {
    case 1:
      return (1.0 / (1.0 + exp(-input)));
    case 2:
      if (input <= 0)
        return 0;
      else
        return input;
    case 3:
      float expInput = exp(input);
      float negExpInput = exp(-input);
      return (expInput - negExpInput) / (expInput + negExpInput);
    case 4:
      return 0;
    default:
      return input;
  }
}
