import torch as t
import torch.nn as nn

from torch import Tensor

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

class ReLU(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, input : Tensor) -> Tensor:
        return t.maximum(input, 0.0)