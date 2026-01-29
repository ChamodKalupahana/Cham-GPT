import torch as t
import torch.nn as nn

from torch import Tensor

from linear import Linear
from relu import ReLU

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

class MLP(nn.Module):
    def __init__(self, in_feats : int, num_hidden_layers : int, out_feats : int):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers

        self.hidden_layer = Linear(in_feats, num_hidden_layers)
        self.output_layer = Linear(num_hidden_layers, out_feats) 
        self.relu = ReLU()

    def forward(self, input: Tensor) -> Tensor:
        hidden = self.hidden_layer(input)
        output = self.output_layer(hidden)
        return self.relu(output)

if __name__ == "__main__":
    in_feats = 100
    out_feats = 20
    hidden = in_feats * 2

    mlp = MLP(in_feats, hidden, out_feats)

    input = t.randn(5, in_feats)
    output = mlp(input)
    print(output)
    print(output.shape)