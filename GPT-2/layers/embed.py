import torch
import torch.nn as nn

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

class Embed(nn.Module):
    def __init__(self, cfg):
        # self.W_E = nn.Parameter()
        return
    
    def forward(self, input):
        return 
    