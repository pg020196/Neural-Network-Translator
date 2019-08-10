from keras.models import load_model
from keras.models import model_from_json
import os
import argparse

parser = argparse.ArgumentParser(description='Decodes .hdf5-files into a .json-file.')
parser.add_argument('-i', '--input', type=str, help='filepath of input .hdf5/h5-file')
parser.add_argument('-o', '--output', type=str, help='filepath of output .json-file', default='model.json')
args = parser.parse_args()
print(args)

model = load_model(args.input) 
model_json = model.to_json()
with open(args.output, "w") as json_file:
    json_file.write(model_json)