from keras.models import load_model
from keras.models import model_from_json
import os


model = load_model('extraordinarilySmartNetwork.h5') 
model_json = model.to_json()
with open("extraordinarilySmartNetworkmodel.json", "w") as json_file:
    json_file.write(model_json)