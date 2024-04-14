import torch

example = torch.tensor([[ 0.1034, -0.1750],
        [ 0.1617, -0.1541],
        [-0.0806,  0.0104],
        [ 0.1600, -0.1236],
        [-0.1364, -0.0308]],)

print(example.argmax(dim=1))