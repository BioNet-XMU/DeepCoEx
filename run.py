import os
import argparse

import torch.cuda

from DeepION import DeepION_training,DeepION_predicting,DimensionalityReduction

import numpy as np

parser = argparse.ArgumentParser(
    description='DeepION for ion image representation of mass spectrometry imaging')

parser.add_argument('--input_Matrix',required= True,help = 'path to inputting MSI data matrix')
parser.add_argument('--input_PeakList',required= False,help = 'path to inputting MSI peak list')
parser.add_argument('--input_shape',required= True,type = int, nargs = '+', help='inputting MSI file shape')
parser.add_argument('--mode',
                    help = 'COL mode for co-localized ion searching',
                    default= 'COL')
parser.add_argument('--ion_mode',
                    help = 'positive or negative ion mode that MSI experiment used',
                    required = False)
parser.add_argument('--tensor_path',
                    help = 'neg_tensor to data augment',
                    required = True)
if __name__ == '__main__':



    #torch.cuda.empty_cache()
    args = parser.parse_args()

    DeepION_training(args.input_Matrix, args.input_shape,args.mode,args.tensor_path)

    features = DeepION_predicting(args.input_Matrix, args.input_shape,args.mode)

    LD_features = DimensionalityReduction(features)

    np.savetxt("DeepCoEx_embedding_20d.txt",LD_features)  #this is the embedding
    




