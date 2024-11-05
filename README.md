## PENet_Steganalysis

This is the PyTorch implementation of the paper "Color Image Steganalysis Based on Pixel Difference Convolution and Enhanced Transformer With Selective Pooling", TIFS 2024. 

## Requirements:
CUDA (10.2)
cuDNN (7.4.1)
python (3.6.9)

## Use
"PENet_Fixed.py" and "PENet_Arbitrary.py" are the main program for fixed-size color images and arbitrary-size color images, respectively. 

"srm_filter_kernel.py" contains the 30 SRM filters. 


For instance: run "PENet_Arbitrary.py" 

Command: python3 PENet_Arbitrary.py -alg HILL-CMDC -rate 0.4 -g 1,2
