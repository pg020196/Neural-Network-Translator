from keras.models import load_model
import numpy as np
from collections import OrderedDict
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import time

#? Loading model and specifying input data
model = load_model('diabetes_model.h5')
data = np.array([[6,148,72,35,0,33.6,0.627,50]])

#? Alternative way of loading the model and building a nn for each layer in order to debug through the nn
# model = Sequential()
# model.add(Dense(8, input_dim=8, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.load_weights('diabetes_model.h5')

# weights_biases_dense_1 = model.layers[0].get_weights()
# weights_biases_dense_2 = model.layers[1].get_weights()

# model_dense_1 = Sequential()
# model_dense_1.add(Dense(8, input_dim=8, activation='sigmoid'))
# model_dense_1.set_weights(weights_biases_dense_1)
# print(weights_biases_dense_1)

# output_dense_1 = model_dense_1.predict(data)
# print(output_dense_1)

# model_dense_2 = Sequential()
# model_dense_2.add(Dense(1, input_dim=8, activation='sigmoid'))
# model_dense_2.set_weights(weights_biases_dense_2)
# print(weights_biases_dense_2)

# ouptut_dense_2 = model_dense_2.predict(output_dense_1)
# print(ouptut_dense_2)

#? Predict and measure the processing time
before = int(round(time.time_ns() / 1000))
output = model.predict(data)
after = int(round(time.time_ns() / 1000))

print("process time in microseconds: " + str(after - before))
print (output)



