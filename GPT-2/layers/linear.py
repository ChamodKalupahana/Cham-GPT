import torch as t
import torch.nn as nn

from torch import Tensor

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

class Linear(nn.Module):
    """
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Applies an affine linear transformation to the incoming data: 
    y = xA.T + b
.
    """
    def __init__(self, in_feats : int, out_feats : int, bias : bool = True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self._use_bias = bias

        weights = t.randn(in_feats, out_feats)
        self.weight = nn.Parameter(weights, requires_grad=True)

        if bias:
            bias_shape = t.randn(out_feats)
            self.bias = nn.Parameter(bias_shape, requires_grad=True)
        else:
            self.bias = None
        return

    def forward(self, input : Tensor) -> Tensor:
        output = input @ self.weight
        if self.bias is not None:
            output += self.bias
        return output


if __name__ == "__main__":
    in_feats = 100
    out_feats = 20

    input = t.rand(60, in_feats)
    linear = Linear(in_feats, out_feats)
    output = linear(input)
    print(output)