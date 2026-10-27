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
mlp_hidden_dimension=64
learning_rate=3e-4
epochs=5
num_channels=1


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed= nn.Conv2d(num_channels, token_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,x):
        x= self.patch_embed(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x 



