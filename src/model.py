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
batch_size=64

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed= nn.Conv2d(num_channels, token_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,x):
        x= self.patch_embed(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x 

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1=nn.LayerNorm(token_dim)
        self.layer_norm2=nn.LayerNorm(token_dim)
        self.multihead_attention=nn.MultiheadAttention(token_dim,num_heads, batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(token_dim, mlp_hidden_dimension), 
            nn.GELU(), 
            nn.Linear(mlp_hidden_dimension, token_dim)
        )
    def forward(self,x):
        residual1=x
        x=self.layer_norm1(x)
        x= self.multihead_attention(x,x,x)[0]

        x=x+residual1

        residual2=x
        x= self.layer_norm2(x)
        x= self.mlp(x)

        x= x+residual2

        return x

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.layernorm1=nn.LayerNorm(token_dim)
        self.mplhead=nn.Linear(token_dim, num_classes
        )
    def forward(self,x):
        x= self.layernorm1(x)
        x= self.mplhead(x)

        return x 


class VisionTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_embedding=PatchEmbedding()
        self.cls_token=nn.Parameter(torch.randn(1,1,token_dim))
        self.position_embedding=nn.Parameter(torch.randn(1,num_patches+1, token_dim))
        self.transformer_block=nn.Sequential(
            *[TransformerEncoder() for _ in range(transformer_blocks)])
        
        self.mlp_head=MLPHead()

    def forward(self,x):
        x= self.patch_embedding(x)
        num_image_in_current_batch=x.shape[0]
        cls_token=self.cls_token.expand(num_image_in_current_batch,-1,-1)
        x=torch.cat((cls_token,x), dim=1)
        x=x+self.position_embedding

        x=self.transformer_block(x)

        x=self.mlp_head(x)

        x=x[:,0]

        return x
        



