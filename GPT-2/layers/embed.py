import torch
import torch.nn as nn

from torch import Tensor

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

class Embed(nn.Module):
    def __init__(self, num_embeddings : int, embedding_dim : int):
        weights = t.rand(num_embeddings, embedding_dim)
        self.W_E = nn.Parameter(weights, requires_grad=True)
        return
    
    def forward(self, input : Tensor) -> Tensor:
        return 
    