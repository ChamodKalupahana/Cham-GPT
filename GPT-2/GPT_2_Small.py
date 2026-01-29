import torch
import torch.nn as nn

model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4, # oddly not 12
    n_ctx=128,
    d_vocab=1000 # TODO: define true vocab size,
)


class GPT_2_Small(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg

    def forward():
        raise NotImplementedError


