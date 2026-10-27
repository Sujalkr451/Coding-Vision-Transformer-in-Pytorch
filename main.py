import torch 
from .src.model import VisionTransformer
import torch.nn as nn

device= torch.mps.is_available()


model= VisionTransformer().to(device)

optimizer=torch.optim.Adam(model.parameters(), lr=3e-4)
criterion=nn.CrossEntropyLoss()

