# PENet_Steganalysis

This is the PyTorch implementation of the paper "Color Image Steganalysis Based on Pixel Difference Convolution and Enhanced Transformer With Selective Pooling", TIFS 2024. 

# Requirements:
CUDA (10.2)
cuDNN (7.4.1)
python (3.6.9)

# Use
"PENet_Fixed.py" and "PENet_Arbitrary.py" are the main program for fixed-size color images and arbitrary-size color images, respectively. 

"srm_filter_kernel.py" contains the 30 SRM filters. 

Example: 

If you want to detect CMD-C-HILL steganography method at 0.4 bpc (on GPU #1), you can enter following command:

"python3 UCNet_Spatial.py -alg CMDC-HILL -rate 0.4 -g 1"

