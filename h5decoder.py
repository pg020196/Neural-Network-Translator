import h5py
import numpy as np
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Decodes .hdf5-files into value matrices.')
parser.add_argument('-i', '--input', type=str, help='filepath of input .hdf5-file')
args = parser.parse_args()
print(args)

# This class allows to decode .hdf5-files to value matrices
class H5Decoder:

    def __init__(self,filepath):
        # Open .hdf5-file
        self.file = h5py.File(filepath,'r')
        self.data1 = list()
        self.data2 = list()

  
    def visitor_func(self, name, node):
        # Check whether the node is a dataset or a group
        if isinstance(node, h5py.Dataset):
            
            print(name)
            print(node)
            
            dataset = np.array(self.file.get(name))
            self.data1.append((name, dataset.tolist()))
            self.data2.append((node, dataset.tolist()))

            print(dataset)
    
    def decode_hdf5_file(self):
        self.file.visititems(self.visitor_func)
        pd.Series(self.data1).to_json(orient='values', path_or_buf='h5decoder_out_1.json')
        pd.Series(self.data2).to_json(orient='values', path_or_buf='h5decoder_out_2.json')

    
h5_decoder = H5Decoder(args.input)
h5_decoder.decode_hdf5_file()