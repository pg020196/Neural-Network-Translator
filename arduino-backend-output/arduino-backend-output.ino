extern "C"{
#include "backend-config.h"
};

//http://www.utopiamechanicus.com/article/arduino-setup-arrays/


void predict()
{

}

void setup() {

  /*
     Opens the serial port and sets the data rate to 9600 bps
  */
  Serial.begin(9600);

  float input[unitsInLayers[0]] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

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

    //Number of rows in A
    const byte m = 1; //numberOfCurrentUnits;
    //Number of columns in A = Number of Rows in B
    unsigned int p = numberOfPreviousUnits;
    //Number of columns in B
    unsigned int n = numberOfCurrentUnits;
    float out[n * m];
    for (unsigned int i = 0; i < m; i++) {
      for (unsigned int j = 0; j < n; j++) {
        out[n * i + j] = 0;
        if (useBias[currentLayer - 1])
        {
          out[n * i + j] = biases[startIndicesBias[currentLayer - 1] + j];
        }
        for (unsigned int k = 0; k < p; k++)
        {
          out[n * i + j] = out[n * i + j] + input[p * i + k] * weights[(n * k + j) + startIndicesWeights[currentLayer - 1]];
        }
        out[n * i + j] = apply_activation_function(activationFunctions[currentLayer - 1], out[n * i + j]);
      }
    }

    memset(input, 0, numberOfCurrentUnits);

    for (unsigned int i = 0; i < numberOfCurrentUnits; i++)
    {
      input[i] = out[i];
    }
  }
  
  uint32_t endTime = micros();
  Serial.println(input[0], 8);
  Serial.println("\ndone");
  Serial.println(String(endTime - startTime) + " microseconds");
}

void loop()
{
  //  if (Serial.available() > 0)
  //  {
  //    Serial.println("Enter input values:");
  //    String a;
  //    a = Serial.readString();
  //    float ArrayKey[unitsInLayers[0]] = {0};
  //    char *tmp;
  //    int i = 0;
  //    tmp = strtok(&a[0], ",");
  //    while (tmp) {
  //      ArrayKey[i++] = atof(tmp);
  //      tmp = strtok(NULL, ",");
  //    }
  //
  //    //  memset(input, 0, unitsInLayers[0];
  //    for (int j = 0; j < unitsInLayers[0]; j++) {
  //      Serial.println(ArrayKey[j], 8);
  //      //      input[j] = ArrayKey[j];
  //    }
  //  }
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
