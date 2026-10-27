import torch 
import torch.nn as nn 

#variables
image_size=28
patch_size=7 
num_patches=(image_size//patch_size)**2
token_dim= 32 
num_heads=4
transformer_blocks=4
num_classes=10 
