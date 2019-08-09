import h5py
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Decodes .hdf5-files into value matrices.')
parser.add_argument('-f', '--filepath', type=str, help='filepath of .hdf5-file')
args = parser.parse_args()
print(args)

# This class allows to decode .hdf5-files to value matrices
class H5Decoder:

    def __init__(self,filepath):
        # Open .hdf5-file
        self.file = h5py.File(filepath,'r')
  
    def visitor_func(self, name, node):
        # Check whether the node is a dataset or a group
        if isinstance(node, h5py.Dataset):
            
            print(name)
            print(node)
            
            dataset = np.array(self.file.get(name))
            
            print(dataset)
    
    def read_hdf5_file(self):
        self.file.visititems(self.visitor_func)
    
h5_decoder = H5Decoder(args.filepath)
h5_decoder.read_hdf5_file()