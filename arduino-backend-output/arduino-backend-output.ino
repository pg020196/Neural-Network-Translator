//http://www.utopiamechanicus.com/article/arduino-setup-arrays/

void setup() {

  Serial.begin(9600); //opens serial port, sets data rate to 9600 bps

  uint32_t ts1 = millis();

  int numberOfLayers = 3;
  int unitsInLayers[3] = {8, 8, 1};
  int activationFunctions[2] = {3, 3};
  int startIndicesWeights[2] = {0, 64};
  int startIndicesBias[2] = {0, 8};
  bool useBias[2] = {true, true};

  float input[unitsInLayers[0]] =
  {
    6, 148, 72, 35, 0, 33.6, 0.627, 50
  };

  float weights[(64 + 8)] =
  {
    -0.15035194158554077, 1.430967926979065, 0.5391924381256104, -0.3918966054916382, 0.4462871849536896, 0.5552943348884583, 0.7700048089027405, 0.28981146216392517,
    -1.689786672592163, 1.042858362197876, 2.2847230434417725, -1.9677045345306396, -0.06611120700836182, -0.061966150999069214, -0.6471585631370544, 1.4964995384216309,
    0.19608011841773987, -0.6369050741195679, 0.09111300855875015, 0.13803385198116302, -0.3828596770763397, -0.4359494745731354, 0.7322076559066772, -0.049885910004377365,
    0.4382251799106598, 0.7386993169784546, -0.7968388199806213, 0.40169671177864075, 1.0121262073516846, 0.2377474009990692, 0.4582376778125763, -0.26966771483421326,
    0.1630777269601822, -0.869050920009613, -0.4613328278064728, 0.2624562084674835, -0.9985146522521973, -0.6456401944160461, -0.12137659639120102, 1.0982298851013184,
    -0.03326864913105965, 0.6471387147903442, -0.1343340277671814, 0.22081176936626434, 0.854404628276825, -1.670728087425232, -0.16672520339488983, 0.8519517779350281,
    1.2319378852844238, -0.01635606400668621, 0.28598904609680176, -0.34935298562049866, 0.049734167754650116, -1.2383219003677368, -0.7718221545219421, 0.7590272426605225,
    0.3540281057357788, -1.3175017833709717, 1.9483529329299927, -2.1833057403564453, -1.0394269227981567, -0.08081339299678802, -2.6470179557800293, -0.5022009611129761,
    0.4447292983531952, -0.8422787189483643, 0.8797550797462463, -0.8606524467468262, -0.1518070250749588, 0.2564888298511505, -0.13789983093738556, -0.3959729075431824
  };


  float biases[8 + 1] =
  {
    0.4447292983531952, -0.8422787189483643, 0.8797550797462463, -0.8606524467468262, -0.1518070250749588, 0.2564888298511505, -0.13789983093738556, -0.3959729075431824,
    -0.2790255844593048
  };

  // init with 1 since we don't work with the input layer
  for (int currentLayer = 1; currentLayer < numberOfLayers; currentLayer++)
  {
    int numberOfPreviousUnits = unitsInLayers[currentLayer - 1];
    int numberOfCurrentUnits = unitsInLayers[currentLayer];

    //Number of rows in A
    int m = numberOfCurrentUnits;
    //Number of columns in A = Number of Rows in B
    int p = numberOfPreviousUnits;
    //Number of columns in B
    int n = 1;
    float out[n * m];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        // printMatrix(1,p,weights[currentLayer-1][i]);

        // initialize with bias value
        out[n * i + j] = 0;
        if (useBias[currentLayer - 1])
        {
          out[n * i + j] = biases[startIndicesBias[currentLayer - 1] + i];
        }

        for (int k = 0; k < p; k++)
        {
          out[n * i + j] = out[n * i + j] + weights[(p * i + k) + startIndicesWeights[currentLayer - 1]] * input[n * k + j];
        }
        out[n * i + j] = apply_activation_function(activationFunctions[currentLayer - 1], out[n * i + j]);
      }
    }
    
    memset(input, 0, numberOfCurrentUnits);
    
    for (int i = 0; i < numberOfCurrentUnits; i++)
    {
      input[i] = out[i];
    }

    Serial.print("\n====OUTPUT LAYER "+String(currentLayer+1)+" ====");
    printMatrix(1, numberOfCurrentUnits, input);

  }
  uint32_t ts2 = millis();
  Serial.println("\ndone");
  Serial.println(String(ts2 - ts1)+ " ms");
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
        Serial.print(matrix[1 * row + column]);
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
