import torch as t
import torch.nn as nn
import einops

from torch import Tensor

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

class Flatten(nn.Module):
    def __init__(self, start_dim : int = 1, end_dim : int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, input : Tensor) -> Tensor:
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

if __name__ == "__main__":
    flatten = Flatten()

    input = t.randn((2, 2, 2))
    print(input)
    output = flatten(input)
    print(output)
    print(output.shape)