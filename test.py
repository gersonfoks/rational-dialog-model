import torch

from utils.analysis import get_abs_and_relative_positions

tokens = torch.tensor([
    [0,0,0,1],
    [0,0,0,1],
    [0,0,1,1],
    [0,0,1,1],
    [0,1,1,1],
    [0,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])

mask = torch.tensor([
    [1,1,1,1],
    [0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],
])
print(mask.shape)
print(get_abs_and_relative_positions(mask, tokens, 0, batch_first=True))
