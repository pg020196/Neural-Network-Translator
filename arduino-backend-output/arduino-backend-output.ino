extern "C" {
#include "backend-config.h"
};

void setup() {

  /*
     Opens the serial port and sets the data rate to 9600 bps.
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

    /*
          Print the received input string to allow verification.
    */
    Serial.println("Input is: " + inputString);

    /*
       Inits the input float array with zeros.
       The length of the array is determined through the number of units in layer 0 (input layer).
    */
    float input[unitsInLayers[0]] = {0};
    char *tmp;
    int i = 0;
    /*
       Tokenize the string on "," characters.
    */
    tmp = strtok(&inputString[0], ",");
    while (tmp) {
      /*
         Convert a string token to a float and store it in the input array.
      */
      input[i++] = atof(tmp);
      tmp = strtok(NULL, ",");
    }

    /*
      Call the predict function with the input array.
    */
    predict(input);

    Serial.println("\nEnter input values:");
  }
}

void predict(float input[])
{
  /*
       Takes the starting time.
  */
  uint32_t startTime = micros();

  /*
     Loops through each layer. Initial layer of current layer = 1 since layer 0 is the input layer.
  */
  for (unsigned int currentLayer = 1; currentLayer < numberOfLayers; currentLayer++)
  {

    /*
      Init two variables to store the important values and improve readability.
    */
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

    /*
       Init the denominator for the softmax activation function.
       If the softmax activation function is used, the denominator can be calucalted during the matrix calculation and thus
       we do not have to loop through the array twice.
    */
    float softmaxDenominator = 0;

    /*
       Loop through the number of units in the current layer.
    */

    for (unsigned int i = 0; i < n; i++) {
      /*
         If bias is activated for the current layer, replace the initial value with the bias value.
      */
      if (useBias[currentLayer - 1])
      {
        out[i] = biases[startIndicesBias[currentLayer - 1] + i];
      }
      /*
         Loop through the number of previous unit to perform the actual "matrix-multiplication".
      */
      for (unsigned int j = 0; j < p; j++)
      {
        out[i] = out[i] + input[j] * weights[(n * j + i) + startIndicesWeights[currentLayer - 1]];
      }

      /*
        Calculates the softmax denominator. This calculation is only required when using the softmax activation function.
        Therefore, we check if the current layer uses the softmax activation function to improve the runtime if another activation function is used.
      */
      if (activationFunctions[currentLayer - 1] == 4) {
        softmaxDenominator = softmaxDenominator + exp(out[i]);
      }
      /*
        Apply activation function directly to element to prevent having to loop through the array a second time.
        If the activation function is equal to 0, no activation function shall be applied since 0 is defined as the linear activation function and thus does not change the value.
        If the activation function is equal to 4, no activation function shall be applied since 4 is defined as the softmax activation function and thus must be applied after all values are calculated.
      */
      if (activationFunctions[currentLayer - 1] != 0 && activationFunctions[currentLayer - 1] != 4) {
        out[i] = applyActivationFunction(activationFunctions[currentLayer - 1], out[i]);
      }
    }

    /*
       Apply the softmax activation function if necessary
    */
    if (activationFunctions[currentLayer - 1] == 4) {
      applyActivationFunctionSoftmax(out, n, softmaxDenominator);
    }


    /*
       Re-init the input array with the dimenion of the current number of units
       and set initial value to 0
    */
    memset(input, 0, numberOfCurrentUnits);

    /*
      Fill the input array with the acutal calculated values
    */
    for (unsigned int i = 0; i < numberOfCurrentUnits; i++)
    {
      input[i] = out[i];
    }
  }

  /*
     Stop the timer
  */
  uint32_t endTime = micros();
  Serial.println(input[0], 8);
  Serial.println("done");
  Serial.println(String(endTime - startTime) + " microseconds");
}

/*
   Applies the specified activation function for the given input value.
   The activation functions can be called as follows:
   0: Linear
   1: Sigmoid
   2: ReLu
   3: Tanh

   To verify the implementation see:
   https://en.wikipedia.org/wiki/Activation_function
*/
float applyActivationFunction(int actFunc, float value)
{
  switch (actFunc) {
    case 1:
      return (1.0 / (1.0 + exp(-value)));
    case 2:
      if (value <= 0)
        return 0;
      else
        return value;
    case 3:
      float expInput = exp(value);
      float negExpInput = exp(-value);
      return (expInput - negExpInput) / (expInput + negExpInput);
    default:
      return value;
  }
}

/*
  Softmax requires a special implementation since the softmax function is applied on the complete vector.

  To verify the implementation see:
  https://en.wikipedia.org/wiki/Activation_function
*/
void applyActivationFunctionSoftmax(float values[], unsigned int inputLength, float denominator) {
  for (int i = 0; i < inputLength; i++) {
    float divisor = exp(values[i]);
    values[i] = divisor / denominator;
  }
}
