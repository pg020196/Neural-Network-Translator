void setup() {

  Serial.begin(9600); //opens serial port, sets data rate to 9600 bps

  uint32_t ts1 = micros();

  int numberOfLayers = ###numberLayers###;
  int unitsInLayers[###numberLayers###] = ###unitsInLayers###;
  int activationFunctions[###dimNumberLayers###] = ###activationFunctions###;
  int startIndicesWeights[###dimNumberLayers###] = ###indicesWeights###;
  int startIndicesBias[###dimNumberLayers###] = ###indicesBias###;
  bool useBias[###dimNumberLayers###] = ###useBias###;

  float input[unitsInLayers[0]] = ###input_data###;

  float weights[###dimWeights###] = ###weights###;

  float biases[###dimBias###] = ###bias###;

  // init with 1 since we don't work with the input layer
  for (int currentLayer = 1; currentLayer < numberOfLayers; currentLayer++)
  {
    int numberOfPreviousUnits = unitsInLayers[currentLayer - 1];
    int numberOfCurrentUnits = unitsInLayers[currentLayer];

    //Number of rows in A
    int m = 1; //numberOfCurrentUnits;
    //Number of columns in A = Number of Rows in B
    int p = numberOfPreviousUnits;
    //Number of columns in B
    int n = numberOfCurrentUnits;
    float out[n * m];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        out[n * i + j] = 0;
        if (useBias[currentLayer - 1])
        {
          out[n * i + j] = biases[startIndicesBias[currentLayer - 1] + j];
        }
        for (int k = 0; k < p; k++)
        {
          out[n * i + j] = out[n * i + j] + input[p * i + k]*weights[(n * k + j) + startIndicesWeights[currentLayer - 1]];
        }
        out[n * i + j] = apply_activation_function(activationFunctions[currentLayer - 1], out[n * i + j]);
      }
    }

    memset(input, 0, numberOfCurrentUnits);

    for (int i = 0; i < numberOfCurrentUnits; i++)
    {
      input[i] = out[i];
    }
  }
  uint32_t ts2 = micros();
  printMatrix(1, 1, input);
  Serial.println("\ndone");
  Serial.println(String(ts2 - ts1) + " ms");
}

void loop() {}

void printMatrix(int rows, int columns, float matrix[])
{
  Serial.println();

  Serial.println();
  for (int row = 0; row < rows; row++)
  {
    for (int column = 0; column < columns; column++)
    {
      Serial.print(matrix[1 * row + column],6);
      Serial.print("   ");
    };
    Serial.print('\n');
  };
}


float apply_activation_function(int actFunc, float input)
{
  //RELU
  if (actFunc == 0)
  {
    if (input <= 0)
    {
      return 0;
    }
    else
    {
      return input;
    }
  }
  //SIGMOID
  else if (actFunc == 1)
  {
    return (1.0 / (1.0 + exp(-input)));
  }
  else
  {
    return input;
  }
}